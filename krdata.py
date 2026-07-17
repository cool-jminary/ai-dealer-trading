"""
데이터 레이어 — 한국 실데이터(kr_prices.csv, kr_meta.csv) 로딩·유니버스·베타

- load_data()      : 종가 패널(dates x 종목코드) + 메타(종목명·시장·섹터)
- daily_returns()  : 일간 수익률 패널
- market_return()  : 동일가중 시장 프록시(베타 계산용)
- screen_universe(): 특정 시점의 거래 가능한 후보군(상장·최소가격·충분한 이력)
- beta_at()        : 특정 시점 각 종목의 시장 베타 (민감도 한도용)
- volume()/adv()   : 거래량 패널 및 최근 N일 평균거래량(ADV) — 돌파 확인·유동성 규정용

거래량(kr_volume.csv)이 있으면 자동 로드되어 ADV 필터·돌파 확인에 쓰인다.
없으면 관련 로직은 자동으로 생략(그레이스풀 폴백)되어 기존 동작을 그대로 유지한다.
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

# OHLC 패널 캐시 — load_data() 시 채워짐. 파일 없으면 종가로 폴백.
_PANELS = {}


def load_data(prices_csv=None, meta_csv=None):
    prices_csv = prices_csv or os.path.join(HERE, "kr_prices.csv")
    meta_csv = meta_csv or os.path.join(HERE, "kr_meta.csv")
    close = pd.read_csv(prices_csv, index_col=0, parse_dates=True)
    close.columns = [str(c).zfill(6) for c in close.columns]     # 종목코드 6자리
    close = close.sort_index()
    meta = pd.read_csv(meta_csv, dtype={"code": str})
    meta["code"] = meta["code"].str.zfill(6)

    # OHLC·거래량 패널(있으면) 로드 → 종가 인덱스/컬럼에 정렬. 없으면 종가로 폴백.
    global _PANELS
    _PANELS = {"close": close}
    for name, fname in [("open", "kr_open.csv"), ("high", "kr_high.csv"),
                        ("low", "kr_low.csv"), ("volume", "kr_volume.csv")]:
        path = os.path.join(HERE, fname)
        if os.path.exists(path):
            p = pd.read_csv(path, index_col=0, parse_dates=True)
            p.columns = [str(c).zfill(6) for c in p.columns]
            _PANELS[name] = p.reindex(index=close.index, columns=close.columns).sort_index()
    return close, meta


def panel(name):
    """'open'/'high'/'low'/'close' 패널 반환. 없으면 종가로 폴백.
    'volume'은 종가 폴백이 무의미하므로 없으면 None을 반환한다."""
    if name == "volume":
        return _PANELS.get("volume")
    return _PANELS.get(name, _PANELS.get("close"))


def has_ohlc():
    """OHLC(고가·저가)가 실제로 로드됐는지."""
    return "high" in _PANELS and "low" in _PANELS


def has_volume():
    """거래량 패널이 실제로 로드됐는지."""
    return "volume" in _PANELS


def adv(date, cols, window=20):
    """date 기준 과거 window영업일 평균거래량(ADV, 주 단위).
    거래량 패널이 없으면 None. cols가 단일 코드 문자열이면 float, 리스트면 Series 반환.
    * 시장충격/유동성 규정(M/O)과 돌파 확인에 쓰는 공용 유동성 지표."""
    v = _PANELS.get("volume")
    if v is None:
        return None
    single = isinstance(cols, str)
    cols_list = [cols] if single else list(cols)
    vv = v.loc[:date, cols_list].iloc[-window:]
    out = vv.mean()
    return float(out.iloc[0]) if single else out


def daily_returns(close):
    return close.pct_change(fill_method=None)


def market_return(rets):
    """동일가중 시장 수익률(상장 종목 평균) — 베타 산정용 프록시"""
    return rets.mean(axis=1)


# ── 변동성 적응형 하락장 방어 (백테스트·시뮬레이터 공용) ──────────────
#   변동성이 클수록: (1) 더 짧은 이평으로 민감하게 감지 (2) 더 강하게 현금화
#   단, 최근이 상승(급등·반등)이면 현금화하지 않는다.
RISKOFF = dict(
    vol_lookback=20,      # 현재 변동성 산정 창(거래일)
    vol_base=252,         # 평소 변동성 기준(1년)
    vol_hi=1.3,           # 변동성이 평소의 1.3배 초과 = '변동성 급증'
    vol_full=2.5,         # 2.5배 이상 = 전량 현금
    ma_calm=20,           # 변동성 낮을 때 추세 판단 이평
    ma_wild=10,           # 변동성 높을 때(짧게) 추세 판단 이평
    mom_win=10,           # 최근 방향(모멘텀) 확인 창
)


def market_proxy(close):
    """동일가중 시장 지수(레벨) — 국면 판단용."""
    return (1 + close.pct_change(fill_method=None).mean(axis=1)).cumprod()


def riskoff_cash(close, date, cfg=RISKOFF):
    """date 시점의 하락장 방어용 '현금 비율'(0~1) 반환.
    '하락 추세 + 변동성 급증'일 때만 >0. 변동성이 클수록 감지 이평이 짧아지고(민감)
    현금 비율이 커진다(강하게). 단, 최근이 상승(급등·반등)이면 현금화하지 않는다.
    반환 0 = 정상(전량 투자), 1 = 전량 현금."""
    idx = market_proxy(close).loc[:date]
    if len(idx) < cfg["vol_base"] + cfg["vol_lookback"]:
        return 0.0
    r = idx.pct_change(fill_method=None)
    cur_vol = r.iloc[-cfg["vol_lookback"]:].std()
    base_vol = r.iloc[-cfg["vol_base"]:].std()
    if base_vol <= 0 or np.isnan(cur_vol):
        return 0.0
    vol_ratio = cur_vol / base_vol                      # 평소 대비 변동성 배수

    # 변동성이 클수록 짧은 이평으로 추세 판단(민감)
    frac = np.clip((vol_ratio - cfg["vol_hi"]) / (cfg["vol_full"] - cfg["vol_hi"]), 0, 1)
    ma_win = int(round(cfg["ma_calm"] + (cfg["ma_wild"] - cfg["ma_calm"]) * frac))
    ma = idx.iloc[-ma_win:].mean()
    downtrend = idx.iloc[-1] < ma                       # 추세이탈(지수 < 이평)

    # 최근 방향: 급등·반등 중이면 방어 안 함 (현금 확보 금지)
    mom = idx.iloc[-1] / idx.iloc[-(cfg["mom_win"] + 1)] - 1 if len(idx) > cfg["mom_win"] else 0.0
    if mom > 0 or not (downtrend and vol_ratio > cfg["vol_hi"]):
        return 0.0                                       # 상승 중이거나 방어조건 미충족 → 현금화 안 함
    return float(frac)                                   # 하락 + 변동성 급증 → 변동성 비례 현금화


def screen_universe(close, date, *, min_history=252, min_price=1000):
    """date 시점의 거래 가능한 종목: 상장(비결측) + 최소가격 + 충분한 과거 이력"""
    hist = close.loc[:date]
    if len(hist) < min_history:
        return []
    last = hist.iloc[-1]
    hist_count = hist.notna().sum()
    ok = last.notna() & (last >= min_price) & (hist_count >= min_history)
    return list(last.index[ok])


def beta_at(rets, mkt, date, cols, window=120):
    """date 기준 과거 window일로 각 종목의 시장 베타 산정"""
    r = rets.loc[:date, cols].iloc[-window:]
    m = mkt.loc[:date].iloc[-window:]
    var_m = np.nanvar(m.values)
    if var_m == 0 or np.isnan(var_m):
        return pd.Series(1.0, index=cols)
    betas = {}
    mv = m.values
    for c in cols:
        rv = r[c].values
        mask = ~np.isnan(rv) & ~np.isnan(mv)
        if mask.sum() < window // 2:
            betas[c] = 1.0
        else:
            cov = np.cov(rv[mask], mv[mask])[0, 1]
            betas[c] = cov / var_m
    return pd.Series(betas)


if __name__ == "__main__":
    close, meta = load_data()
    print("종가 패널:", close.shape, "| 기간:", close.index.min().date(), "~", close.index.max().date())
    print("메타:", meta.shape, "| 섹터 유효:", meta["sector"].notna().sum(), "/", len(meta))
    rets = daily_returns(close)
    mkt = market_return(rets)
    d = close.index[-1]
    uni = screen_universe(close, d)
    print(f"\n{d.date()} 유니버스: {len(uni)}종목")
    b = beta_at(rets, mkt, d, uni[:5])
    print("샘플 베타:", {k: round(v, 2) for k, v in b.items()})
    print("OHLC 로드:", has_ohlc(), "| 거래량 로드:", has_volume())
    if has_volume():
        a = adv(d, uni[:5])
        print("샘플 ADV(20일):", {k: f"{v:,.0f}주" for k, v in a.items()})