"""
포트폴리오 백테스트 + 지수가중 선정 + 반감기(1/3/6개월) 비교

- 레이어 A(선정 지표): 학습구간 성과를 지수가중한 '가중 샤프'로 1등 선정 (최근 중시)
- 레이어 B(종목 선정): 각 agent가 반감기로 반응 속도 조절 (kr_agents)
- 반감기 1·3·6개월 각각으로 전체를 돌려, (반감기 × 룰) 조합 중 가중 샤프 최고를 채택
- look-ahead 차단 · 거래비용 · 손익(낙폭) 위험회피 · 워크포워드 검증
"""
import numpy as np
import pandas as pd

from krdata import (load_data, daily_returns, market_return, screen_universe,
                    riskoff_cash)
from kr_agents import make_agents

REBAL = 21
COST = 0.0015
WARMUP = 252
HALFLIVES = {"1개월": 21, "3개월": 63, "6개월": 126}

# 공통 리스크 매니저 (모든 agent 위에 일괄 적용되는 매도 규칙)
STOP_LOSS = 0.20       # 진입가 대비 -20% → 손절(변동성 장 대응)
TRAIL_STOP = 0.30      # 보유 중 고점 대비 -30% → 추격손절
RULE_EXIT_EVERY = 5    # 룰이탈 매도 점검 주기(거래일, 주 1회)


def _turnover(prev, new):
    return sum(abs(new.get(c, 0) - prev.get(c, 0)) for c in set(prev) | set(new))


def _vol_panel(close, rets):
    """종목별 일간 변동성(가격 대비 프랙션). OHLC 있으면 ATR/price, 없으면 종가수익률 std."""
    import krdata
    if krdata.has_ohlc():
        high = krdata.panel("high"); low = krdata.panel("low"); pc = close.shift(1)
        tr = np.maximum(high - low, np.maximum((high - pc).abs(), (low - pc).abs()))
        return (tr.rolling(20).mean() / close)          # ATR(20) / 종가
    return rets.rolling(20).std()                        # 종가 수익률 20일 표준편차


def _run_buyhold_bt(agent, close, rets, mkt, meta):
    """Buy & Hold 전용 엔진 — 최초 1회 매수 후 리밸런싱·매도 없음.
    보유 비중은 가격에 따라 자연 성장(승자 확대). 피라미딩(init_invest<1)이면
    진입가 대비 add_trigger마다 남은 현금으로 추가 매수(max_adds회)."""
    dates = close.index
    held, entry, adds = {}, {}, {}
    cash_w, per0 = 1.0, 0.0
    out_dates, out_rets = [], []

    for i in range(WARMUP, len(dates) - 1):
        date = dates[i]
        cost = 0.0

        if not held:                                   # ── 최초 1회 매수
            uni = screen_universe(close, date)
            target = agent.construct(close, rets, mkt, meta, date, uni)
            if target:
                scale = agent.init_invest / sum(target.values())
                held = {c: w * scale for c, w in target.items()}
                per0 = agent.init_invest / len(held)    # 종목당 초기 비중
                for c in held:
                    entry[c] = float(close.at[date, c]); adds[c] = 0
                cash_w = 1.0 - sum(held.values())
                cost += COST * sum(held.values())

        elif agent.max_adds > 0 and cash_w > 1e-9:     # ── 피라미딩: 수익 종목 추가매수
            for c in list(held):
                if adds[c] >= agent.max_adds or cash_w <= 1e-9:
                    continue
                p = float(close.at[date, c])
                if np.isnan(p) or entry[c] <= 0:
                    continue
                if p / entry[c] - 1 >= agent.add_trigger:     # 진입가 대비 +트리거%
                    add = min(per0 * 0.5, cash_w)             # 초기비중의 절반씩
                    if add <= 1e-9:
                        continue
                    held[c] += add; cash_w -= add
                    cost += COST * add
                    entry[c] = p                              # 기준가 재설정(다음 트리거)
                    adds[c] += 1

        # 일간 수익 (date → nxt) + 비중 자연 성장
        nxt = dates[i + 1]
        r = rets.loc[nxt].reindex(held.keys()).fillna(0.0) if held else pd.Series(dtype=float)
        grown = {c: held[c] * (1 + float(r.get(c, 0.0))) for c in held}
        total = sum(grown.values()) + cash_w
        day_ret = (total - 1.0) - cost
        if total > 0:
            held = {c: v / total for c, v in grown.items()}   # 재정규화(비중 = 평가액 비율)
            cash_w = cash_w / total
        out_dates.append(nxt); out_rets.append(day_ret)

    return (1 + pd.Series(out_rets, index=out_dates)).cumprod()


def run_portfolio_bt(agent, close, rets, mkt, meta,
                     dynamic_stop=False, k_stop=5, k_trail=8, defend=False):
    """월간 리밸런싱 + 3층 매도. dynamic_stop=True면 종목별 변동성 비례 손절.
    defend=True면 변동성 적응형 하락장 방어(현금화) 적용.
    agent.buy_and_hold=True면 전용 엔진(매수 후 보유)으로 분기."""
    if getattr(agent, "buy_and_hold", False):
        return _run_buyhold_bt(agent, close, rets, mkt, meta)
    dates = close.index
    vol = _vol_panel(close, rets) if dynamic_stop else None   # 종목별 변동성(ATR 또는 종가std)
    held, pos = {}, {}          # held: code->weight,  pos: code->{entry,peak}
    eq, risk_off = 1.0, False
    eq_hist = [1.0]
    out_dates, out_rets = [], []

    for i in range(WARMUP, len(dates) - 1):
        date = dates[i]
        cost = 0.0

        # 보유 종목 고점 갱신
        for c in held:
            p = float(close.at[date, c])
            if not np.isnan(p):
                pos[c]["peak"] = max(pos[c]["peak"], p)

        if (i - WARMUP) % REBAL == 0:                      # ── 월간 리밸런싱
            risk_off = False
            uni = screen_universe(close, date)
            target = agent.construct(close, rets, mkt, meta, date, uni)
            cost += COST * _turnover(held, target)
            newpos = {}
            for c in target:
                p = float(close.at[date, c])
                newpos[c] = pos.get(c, {"entry": p, "peak": p})
            held, pos = dict(target), newpos
        else:                                              # ── 매도 규칙 점검
            sells = set()
            for c in list(held):                           # 공통 손절·트레일링(agent별 적용)
                p = float(close.at[date, c])
                if np.isnan(p):
                    continue
                if dynamic_stop:                           # 변동성 비례(ATR식) 유동 손절
                    v = vol.at[date, c]
                    if np.isnan(v):
                        v = 0.02
                    stop_w = min(max(k_stop * v, 0.08), 0.35)    # 손절폭 8~35%
                    trail_w = min(max(k_trail * v, 0.12), 0.45)  # 트레일폭 12~45%
                else:                                      # 고정 손절
                    stop_w, trail_w = STOP_LOSS, TRAIL_STOP
                if agent.use_stop and p <= pos[c]["entry"] * (1 - stop_w):
                    sells.add(c)
                elif agent.use_trail and p <= pos[c]["peak"] * (1 - trail_w):
                    sells.add(c)
            if held and (i - WARMUP) % RULE_EXIT_EVERY == 0:   # 룰이탈(주 1회)
                em = agent.exit_mask(close, date, list(held))
                sells |= set(em.index[em.fillna(False)])
            for c in sells:
                cost += COST * held[c]
                held.pop(c, None); pos.pop(c, None)

        # 일간 수익 (date → nxt) — 하락장이면 현금 비율만큼 노출 축소
        nxt = dates[i + 1]
        cash = riskoff_cash(close, date) if defend else 0.0   # 0=정상, 1=전량현금
        if held and not risk_off:
            w = pd.Series(held)
            r = rets.loc[nxt].reindex(w.index).fillna(0.0)
            day_ret = float((w * r).sum()) * (1 - cash) - cost   # 현금분은 수익 0
        else:
            day_ret = -cost

        eq *= (1 + day_ret)
        eq_hist.append(eq)
        trail_peak = max(eq_hist[-252:])              # 최근 1년 고점 기준(사상최고 아님)
        if eq / trail_peak - 1 <= -agent.limits.max_drawdown_limit:   # 트레일링 낙폭 정지
            risk_off = True; held, pos = {}, {}       # 다음 리밸런싱까지 현금 → 회복 시 재편입
        out_dates.append(nxt); out_rets.append(day_ret)

    return (1 + pd.Series(out_rets, index=out_dates)).cumprod()


def perf(eq):
    eq = eq / eq.iloc[0]
    r = eq.pct_change(fill_method=None).dropna(); days = len(r)
    total = eq.iloc[-1] - 1
    cagr = (1 + total) ** (252 / days) - 1 if days else 0.0
    vol = r.std() * np.sqrt(252)
    sharpe = cagr / vol if vol and vol > 0 else 0.0
    mdd = (eq / eq.cummax() - 1).min()
    return dict(total=total, cagr=cagr, vol=vol, sharpe=sharpe, mdd=mdd)


def recency_weighted_sharpe(rets, halflife):
    """학습구간 일수익률을 지수가중(최근 큰 가중)한 샤프 — 선정 지표"""
    r = rets.dropna().values; n = len(r)
    if n < 20:
        return 0.0
    w = 0.5 ** (np.arange(n)[::-1] / halflife)
    wm = np.sum(w * r) / np.sum(w)
    wv = np.sum(w * (r - wm) ** 2) / np.sum(w)
    return (wm / np.sqrt(wv)) * np.sqrt(252) if wv > 0 else 0.0


# 하위호환: 기존 오케스트레이터가 쓰는 단순 워크포워드(가중 없음)
def walk_forward_select(agents, close, rets, mkt, meta, train_frac=0.6):
    curves = {ag.name: run_portfolio_bt(ag, close, rets, mkt, meta) for ag in agents}
    anyc = next(iter(curves.values())); split = anyc.index[int(len(anyc) * train_frac)]
    rows = {n: {"train": perf(eq.loc[:split]), "valid": perf(eq.loc[split:])} for n, eq in curves.items()}
    winner = max(rows, key=lambda k: rows[k]["train"]["sharpe"])
    valid_best = max(rows, key=lambda k: rows[k]["valid"]["sharpe"])
    return curves, rows, winner, valid_best, split


def select_with_halflife(close, rets, mkt, meta, train_frac=0.6, split_date=None):
    results = {}
    for label, H in HALFLIVES.items():
        curves = {ag.name: run_portfolio_bt(ag, close, rets, mkt, meta) for ag in make_agents(H)}
        anyc = next(iter(curves.values()))
        if split_date is not None:
            split = anyc.index[anyc.index.get_indexer([pd.Timestamp(split_date)], method="nearest")[0]]
        else:
            split = anyc.index[int(len(anyc) * train_frac)]
        rows = {}
        for n, eq in curves.items():
            train_ret = eq.loc[:split].pct_change(fill_method=None).dropna()
            rows[n] = {"wsharpe": recency_weighted_sharpe(train_ret, H),
                       "train": perf(eq.loc[:split]), "valid": perf(eq.loc[split:]), "curve": eq}
        winner = max(rows, key=lambda k: rows[k]["wsharpe"])
        results[label] = {"H": H, "rows": rows, "winner": winner, "split": split,
                          "wsharpe": rows[winner]["wsharpe"]}
    best_hl = max(results, key=lambda L: results[L]["wsharpe"])
    return results, best_hl


def pct(x): return f"{x*100:6.1f}%"


if __name__ == "__main__":
    close, meta = load_data()
    rets = daily_returns(close); mkt = market_return(rets)
    results, best_hl = select_with_halflife(close, rets, mkt, meta, train_frac=0.6)

    split = results[best_hl]["split"]
    print(f"학습/검증 분할일: {split.date()}   |   선정 지표: 지수가중 샤프(최근 중시)\n")
    hdr = f"{'반감기':>6} {'agent':>15} | {'가중샤프(학습)':>13} | {'검증샤프':>8}{'검증수익':>9}{'검증MDD':>9}"
    print(hdr); print("-" * len(hdr))
    for label, res in results.items():
        for n, r in res["rows"].items():
            star = " ★" if n == res["winner"] else "  "
            hl = label if n == res["winner"] else ""
            print(f"{hl:>6} {n:>15}{star}|{r['wsharpe']:>13.2f} |"
                  f"{r['valid']['sharpe']:>8.2f}{pct(r['valid']['total']):>9}{pct(r['valid']['mdd']):>9}")
        print("-" * len(hdr))
    win = results[best_hl]
    print(f"\n→ 최종 채택: 반감기 {best_hl} · {win['winner']}  (가중샤프 {win['wsharpe']:.2f})")
    print("   반감기별 1등 가중샤프:", {L: round(results[L]["wsharpe"], 2) for L in results})

    # 차트: 반감기별 우승 곡선 + 시장
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        colors = {"1개월": "#E4685F", "3개월": "#1D9E75", "6개월": "#5B9BD6"}
        fig, ax = plt.subplots(figsize=(11, 5.4))
        for label, res in results.items():
            eq = res["rows"][res["winner"]]["curve"]; eq = eq / eq.iloc[0]
            lw = 3.0 if label == best_hl else 1.5
            ax.plot(eq.index, eq.values, color=colors[label], linewidth=lw,
                    label=f"HL {label}: {res['winner']}" + (" (채택)" if label == best_hl else ""))
        idx = res["rows"][res["winner"]]["curve"].index
        bench = (1 + mkt.loc[idx]).cumprod(); bench = bench / bench.iloc[0]
        ax.plot(bench.index, bench.values, "--", color="#888780", linewidth=1.1, label="market (eq-weight)")
        ax.axvline(split, color="#888780", linewidth=1, linestyle=":")
        ax.text(split, ax.get_ylim()[1], " validation", va="top", fontsize=9, color="#5F5E5A")
        ax.set_title("Half-life comparison — winner per half-life (recency-weighted selection)", fontsize=12.5)
        ax.set_ylabel("Normalized equity (start = 1.0)")
        ax.legend(loc="upper left", frameon=False, fontsize=9); ax.grid(alpha=0.25)
        fig.tight_layout(); fig.savefig("kr_halflife_result.png", dpi=140)
        print("\n차트 저장: kr_halflife_result.png")
    except Exception as e:
        print(f"\n(차트 생략: {e})")