"""
포트폴리오 Agent — 각 딜링 기법으로 '종목 선정' (cross-sectional) + 반감기 파라미터

반감기(halflife)는 각 룰의 '반응 속도' 노브다. 짧을수록 최근 움직임을 더 본다.
  - 추세추종  : 지수가중 모멘텀 (최근 수익률에 큰 가중)
  - 평균회귀  : 지수가중 기준선(EWMA) 대비 과매도
  - 돌파      : 채널 길이를 반감기에 연동 (짧을수록 빠른 돌파 포착)
               + 거래량 확인(당일 거래량 ≥ vol_mult×ADV)으로 가짜 돌파(whipsaw) 필터
               ※ 거래량 데이터 없으면 확인 조건 자동 생략(그레이스풀 폴백)

한도:
  - 리스크 : 단일 종목 비중 상한, 총 노출(gross) 상한
  - 민감도 : 포트폴리오 시장 베타 상한 → 초과 시 노출 축소
  - (섹터) : 섹터 집중도 상한 (메타에 섹터가 있을 때만)
"""
from dataclasses import dataclass, field
import os
import numpy as np
import pandas as pd

from krdata import beta_at

_SHARES_CACHE = None


def _shares():
    """상장주식수 (kr_shares.csv) 로드 — 시총 = 주가 x 상장주식수"""
    global _SHARES_CACHE
    if _SHARES_CACHE is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kr_shares.csv")
        s = pd.read_csv(path, dtype={"code": str})
        s["code"] = s["code"].str.zfill(6)
        _SHARES_CACHE = s.set_index("code")["shares"].astype(float)
    return _SHARES_CACHE


def _rsi_last(c: pd.DataFrame, window=14) -> pd.Series:
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(window).mean().iloc[-1]
    loss = (-delta.clip(upper=0)).rolling(window).mean().iloc[-1]
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


@dataclass
class PortfolioLimits:
    top_n: int = 15
    max_position_weight: float = 0.12
    target_gross: float = 1.0
    max_beta: float = 1.2
    max_sector_weight: float = 0.35
    max_drawdown_limit: float = 0.30  # 손익: 트레일링 고점 대비 -30% → 전량 청산


@dataclass
class BasePortfolioAgent:
    name: str = "agent"
    halflife: int = 63                 # 반감기(거래일). 63≈3개월
    limits: PortfolioLimits = field(default_factory=PortfolioLimits)
    min_score: float = 0.0
    use_stop: bool = True              # 공통 손절(-8%) 적용 여부
    use_trail: bool = True             # 공통 트레일링(-12%) 적용 여부
    buy_and_hold: bool = False         # True면 최초 1회 매수 후 리밸런싱·매도 없음

    def rank(self, close, date, universe) -> pd.Series:
        raise NotImplementedError

    def construct(self, close, rets, mkt, meta, date, universe) -> dict:
        scores = self.rank(close, date, universe).dropna()
        scores = scores[scores > self.min_score]
        if scores.empty:
            return {}
        picks = scores.sort_values(ascending=False).head(self.limits.top_n)
        w = min(self.limits.target_gross / len(picks), self.limits.max_position_weight)
        weights = {c: w for c in picks.index}
        weights = self._sector_cap(weights, picks, meta)
        weights = self._beta_cap(weights, rets, mkt, date)
        return weights

    def _sector_cap(self, weights, picks, meta):
        smap = meta.set_index("code")["sector"]
        if smap.notna().sum() == 0:
            return weights
        changed = True
        while changed and weights:
            changed = False
            by_sec = {}
            for c in weights:
                by_sec.setdefault(smap.get(c, "N/A"), []).append(c)
            total = sum(weights.values())
            for s, members in by_sec.items():
                if sum(weights[c] for c in members) / max(total, 1e-9) > self.limits.max_sector_weight and len(members) > 1:
                    del weights[min(members, key=lambda c: picks.get(c, 0))]
                    changed = True
                    break
        return weights

    def _beta_cap(self, weights, rets, mkt, date):
        if not weights:
            return weights
        betas = beta_at(rets, mkt, date, list(weights.keys()))
        port_beta = sum(weights[c] * betas.get(c, 1.0) for c in weights)
        if port_beta > self.limits.max_beta:
            scale = self.limits.max_beta / port_beta
            weights = {c: v * scale for c, v in weights.items()}
        return weights

    # 룰이탈 매도: 보유 종목이 더 이상 이 룰에 맞지 않으면 청산 (기본: 청산 안 함)
    def exit_mask(self, close, date, codes):
        return pd.Series(False, index=codes)


# ---- 1) 추세추종: 지수가중 모멘텀 ------------------------------------
@dataclass
class MomentumAgent(BasePortfolioAgent):
    name: str = "momentum"

    def rank(self, close, date, universe):
        c = close.loc[:date, universe]
        if len(c) < 252:
            return pd.Series(dtype=float)
        r = c.pct_change(fill_method=None).iloc[-252:]
        w = 0.5 ** (np.arange(len(r))[::-1] / self.halflife)   # 최근일수록 큰 가중
        return r.mul(w, axis=0).sum() / w.sum()                # 가중 평균 일수익률

    def exit_mask(self, close, date, codes):
        c = close.loc[:date, codes]
        r = c.pct_change(fill_method=None).iloc[-252:]
        w = 0.5 ** (np.arange(len(r))[::-1] / self.halflife)
        score = r.mul(w, axis=0).sum() / w.sum()
        return score <= 0                                      # 추세 소멸 → 청산


# ---- 2) 평균회귀: 지수가중 기준선(EWMA) 대비 과매도 ------------------
@dataclass
class MeanReversionAgent(BasePortfolioAgent):
    name: str = "mean_reversion"
    use_trail: bool = False            # 되돌림 전략: 트레일링 미적용(손절은 유지)

    def rank(self, close, date, universe):
        c = close.loc[:date, universe]
        if len(c) < 200:
            return pd.Series(dtype=float)
        price = c.iloc[-1]
        ewm_ref = c.ewm(halflife=self.halflife).mean().iloc[-1]  # 반응형 기준선
        ma_long = c.iloc[-200:].mean()
        rsi = _rsi_last(c, 14)
        below = ewm_ref / price - 1                             # >0: 기준선 아래(과매도)
        eligible = (price > ma_long) & (below > 0) & (rsi < 45)
        return below.where(eligible)

    def exit_mask(self, close, date, codes):
        c = close.loc[:date, codes]
        price = c.iloc[-1]
        ewm_ref = c.ewm(halflife=self.halflife).mean().iloc[-1]
        rsi = _rsi_last(c, 14)
        return (price >= ewm_ref) | (rsi >= 55)                # 평균 복귀 → 청산


# ---- 3) 돌파: 채널 길이를 반감기에 연동 -------------------------------
@dataclass
class BreakoutAgent(BasePortfolioAgent):
    name: str = "breakout"
    vol_mult: float = 1.5              # 돌파 확인: 당일 거래량 ≥ vol_mult × ADV(vol_window)
    vol_window: int = 20              # ADV 산정 기간(거래일)

    def rank(self, close, date, universe):
        import krdata
        window = max(20, int(round(self.halflife)))            # 반감기=채널 길이
        c = close.loc[:date, universe]
        if len(c) < window + 1:
            return pd.Series(dtype=float)
        hp = krdata.panel("high")
        high = (hp if hp is not None else close).loc[:date, universe]  # OHLC 없으면 종가
        price = c.iloc[-1]                                     # 현재 종가
        cur_high = high.iloc[-1]                               # 당일 고가(장중)
        prior_high = high.iloc[-(window + 1):-1].max()         # 직전 N일 고가 채널
        mom20 = price / c.iloc[-21] - 1
        broke = cur_high >= prior_high                         # 장중 고가가 채널 돌파

        # 거래량 확인: 거래량 실린 돌파만 인정(가짜 돌파 필터). 데이터 없으면 생략.
        vp = krdata.panel("volume")
        if vp is not None:
            vol = vp.loc[:date, universe]
            today_vol = vol.iloc[-1]
            adv = vol.iloc[-(self.vol_window + 1):-1].mean()   # 당일 제외 직전 평균거래량
            vol_ok = today_vol >= self.vol_mult * adv          # 평소 대비 거래량 급증 동반
            broke = broke & vol_ok.reindex(broke.index).fillna(False)
        return mom20.where(broke)                              # 돌파 종목을 20일 모멘텀으로 랭킹

    def exit_mask(self, close, date, codes):
        import krdata
        c = close.loc[:date, codes]
        if len(c) < 11:
            return pd.Series(False, index=codes)
        lp = krdata.panel("low")
        low = (lp if lp is not None else close).loc[:date, codes]      # OHLC 없으면 종가
        price = c.iloc[-1]
        low10 = low.iloc[-11:-1].min()                         # 10일 저가 채널 하단
        return price < low10                                   # 채널 하단 이탈 → 청산


# ---- 4) 시총 상위 10 바스켓 (대형주 추종, ≈패시브) --------------------
@dataclass
class TopCapBasketAgent(BasePortfolioAgent):
    name: str = "top_cap"
    limits: PortfolioLimits = field(default_factory=lambda: PortfolioLimits(top_n=10))
    use_stop: bool = False             # 대형주 바스켓: 조정을 견딤(자체 규칙=10위 이탈)
    use_trail: bool = False

    def _caps(self, close, date):
        sh = _shares()
        px = close.loc[date]
        common = sh.index.intersection(px.index)
        return (px[common] * sh[common]).dropna()              # 시가총액

    def rank(self, close, date, universe):
        caps = self._caps(close, date).reindex(universe).dropna()
        return caps                                            # 시총 큰 순 → 상위 10

    def exit_mask(self, close, date, codes):
        caps = self._caps(close, date)
        top = set(caps.sort_values(ascending=False).head(self.limits.top_n).index)
        return pd.Series([c not in top for c in codes], index=codes)  # 10위 이탈 → 청산


# ---- 5) 저변동성 (Low Volatility) — 변동성 낮은 종목 선호(방어형) ----------
@dataclass
class LowVolAgent(BasePortfolioAgent):
    name: str = "low_vol"
    use_trail: bool = False            # 방어형: 저변동 유지, 트레일링 미적용
    vol_window: int = 60

    def rank(self, close, date, universe):
        c = close.loc[:date, universe]
        if len(c) < self.vol_window + 5:
            return pd.Series(dtype=float)
        vol = c.pct_change(fill_method=None).iloc[-self.vol_window:].std()
        ma = c.iloc[-120:].mean() if len(c) >= 120 else c.mean()
        price = c.iloc[-1]
        inv = 1.0 / vol.replace(0, np.nan)                # 저변동일수록 높은 점수
        return inv.where(price >= ma).dropna()            # 급락 종목 제외(장기이평 위)

    def exit_mask(self, close, date, codes):
        c = close.loc[:date, codes]
        if len(c) < 120:
            return pd.Series(False, index=codes)
        return c.iloc[-1] < c.iloc[-120:].mean()          # 장기이평 이탈 시 청산


# ---- 6) 52주 신고가 근접 (George & Hwang) --------------------------------
@dataclass
class FiftyTwoWeekHighAgent(BasePortfolioAgent):
    name: str = "week52_high"

    def rank(self, close, date, universe):
        c = close.loc[:date, universe]
        if len(c) < 252:
            return pd.Series(dtype=float)
        prox = c.iloc[-1] / c.iloc[-252:].max()           # 1.0에 가까울수록 신고가 근접
        return prox.where(prox >= 0.90).dropna()

    def exit_mask(self, close, date, codes):
        c = close.loc[:date, codes]
        if len(c) < 252:
            return pd.Series(False, index=codes)
        return (c.iloc[-1] / c.iloc[-252:].max()) < 0.80  # 고점 대비 -20% 이탈


# ---- 7) 거래량 기반 (가격·거래량 동반 상승) — kr_volume.csv 필요 -----------
@dataclass
class VolumeFlowAgent(BasePortfolioAgent):
    name: str = "volume_flow"
    vol_window: int = 20

    def rank(self, close, date, universe):
        import krdata
        vp = krdata.panel("volume")
        if vp is None:                                    # 거래량 없으면 이 봇은 선택 없음
            return pd.Series(dtype=float)
        c = close.loc[:date, universe]
        if len(c) < self.vol_window * 4:
            return pd.Series(dtype=float)
        vol = vp.loc[:date, universe]
        mom20 = c.iloc[-1] / c.iloc[-21] - 1                          # 가격 모멘텀
        recent = vol.iloc[-self.vol_window:].mean()
        base = vol.iloc[-(self.vol_window*4):-self.vol_window].mean()
        surge = recent / base.replace(0, np.nan)                     # 거래량 증가율
        ok = (mom20 > 0) & (surge > 1.0)
        return (mom20 * surge).where(ok).dropna()                    # 가격↑ + 거래량↑

    def exit_mask(self, close, date, codes):
        c = close.loc[:date, codes]
        return (c.iloc[-1] / c.iloc[-21] - 1) <= 0                   # 모멘텀 소멸 시 청산


# ---- 8) 순수 Buy & Hold — 시총 상위 20 사서 안 판다 -----------------------
@dataclass
class BuyHoldAgent(BasePortfolioAgent):
    """최초 1회 시총 상위 20종목 동일가중 매수 후 리밸런싱·매도 없음.
    오른 종목의 비중이 자연히 커진다(승자 자동 확대). 액티브 전략들의 기준선."""
    name: str = "buy_hold"
    limits: PortfolioLimits = field(default_factory=lambda: PortfolioLimits(
        top_n=20, max_position_weight=0.20, max_beta=99.0, max_sector_weight=1.0))
    use_stop: bool = False
    use_trail: bool = False
    buy_and_hold: bool = True
    init_invest: float = 1.00          # 전액 투자 (현금 없음)
    add_trigger: float = 0.0           # 추가매수 없음
    max_adds: int = 0

    def rank(self, close, date, universe):
        sh = _shares()
        px = close.loc[date, universe].dropna()
        common = px.index.intersection(sh.index)
        return (px[common] * sh[common]).dropna()      # 시가총액 상위 20


# ---- 9) 피라미딩 Buy & Hold — 수익 나면 추가 매수 -------------------------
@dataclass
class PyramidBuyHoldAgent(BuyHoldAgent):
    """초기 70% 투자 + 현금 30%. 진입가 대비 +20%마다 추가 매수(최대 3회).
    '수익 나면 더 산다'는 피라미딩. 재원은 남겨둔 현금."""
    name: str = "buy_hold_pyramid"
    init_invest: float = 0.70          # 나머지 30%는 추가매수 재원
    add_trigger: float = 0.20          # +20%마다
    max_adds: int = 3                  # 최대 3회


def make_agents(halflife: int = 63):
    return [MomentumAgent(halflife=halflife),
            MeanReversionAgent(halflife=halflife),
            BreakoutAgent(halflife=halflife),
            TopCapBasketAgent(halflife=halflife),
            LowVolAgent(halflife=halflife),
            FiftyTwoWeekHighAgent(halflife=halflife),
            VolumeFlowAgent(halflife=halflife),
            BuyHoldAgent(halflife=halflife),
            PyramidBuyHoldAgent(halflife=halflife)]


if __name__ == "__main__":
    from krdata import load_data, daily_returns, market_return, screen_universe
    close, meta = load_data()
    rets = daily_returns(close); mkt = market_return(rets)
    date = close.index[-1]; uni = screen_universe(close, date)
    names = meta.set_index("code")["name"]
    for H in (21, 63, 126):
        print(f"\n=== 반감기 {H}일 ===")
        for ag in make_agents(H):
            w = ag.construct(close, rets, mkt, meta, date, uni)
            top = sorted(w.items(), key=lambda x: -x[1])[:3]
            print(f" [{ag.name}] {len(w)}종목:",
                  ", ".join(f"{names.get(c,c)}" for c, _ in top))