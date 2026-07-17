"""
일자 진행형 트레이딩 시뮬레이터 + 운용한도/손실한도 규정 (발표용)

- 운용한도 300억, 매년 1월초 평가액 300억으로 리셋(연 정산). 연도 선택 → 그 해 누적손익
- 2023-01 ~ 2026-05: 월1회 선정으로 사전 운용(성과 표시) → 그 상태로 6월부터 일자 진행
- M/O 손실한도 규정:
    · 일일 손실 평가액의 10%(당일)   → 1주(5영업일) 신규매매 정지 + 알람 (보유 유지)
    · 월간 손실 월초의 30%(월초대비) → 운용한도 평가액의 1/3로 축소(손실 큰 종목 매도) + 1개월(21영업일)
                                 신규매매 정지 + 딜러 메시지. 정지 해제 시 300억 복구
- 매수/매도 종목명+확인, 반려 사유+확인
"""
import os
from flask import Flask, jsonify, request, send_file
import pandas as pd

from krdata import load_data, daily_returns, market_return, screen_universe
import krdata
from kr_agents import make_agents
from kr_backtest import run_portfolio_bt, recency_weighted_sharpe
from mo_engine import check_compliance, HUMAN_VALUE   # HUMAN_VALUE=30억 딜러 결재선
import llm   # OpenAI(ChatGPT 4o); 키 없으면 규칙 기반 폴백

HERE = os.path.dirname(os.path.abspath(__file__))
LIMIT_FULL = 30_000_000_000       # 운용한도 300억 (연초 리셋 기준)
LIMIT_REDUCED = 10_000_000_000    # (참고) 축소 하한
DAILY_LOSS_PCT = 0.10             # 일일 손실한도 = 그날 시작 평가액의 10%
MONTHLY_LOSS_PCT = 0.30           # 월간 손실한도 = 월초 평가액의 30%
REDUCE_FACTOR = 1/3               # 월간 발동 시 운용한도 = 평가액의 1/3
DAILY_FREEZE = 5                  # 5영업일 정지
MONTHLY_FREEZE = 21              # 21영업일 정지
HIST_START = "2023-01-01"
SIM_START = "2026-06-01"
RESELECT_EVERY = 21
HALFLIFE = 63
AGENT_KR = {"momentum": "모멘텀", "mean_reversion": "평균회귀",
            "breakout": "돌파", "top_cap": "시총상위10",
            "low_vol": "저변동성", "week52_high": "52주신고가", "volume_flow": "거래량",
            "buy_hold": "BuyHold순수", "buy_hold_pyramid": "BuyHold피라미딩"}

app = Flask(__name__)
S = {}


def kr_tick(p):
    for lim, t in [(2000,1),(5000,5),(20000,10),(50000,50),(200000,100),(500000,500)]:
        if p < lim: return t
    return 1000


def build_sim():
    close, meta = load_data()
    rets = daily_returns(close); mkt = market_return(rets)
    names = meta.set_index("code")["name"]
    agents = {a.name: a for a in make_agents(HALFLIFE)}
    curves = {n: run_portfolio_bt(a, close, rets, mkt, meta, dynamic_stop=True)
              for n, a in agents.items()}
    dret = pd.DataFrame({n: c.pct_change(fill_method=None) for n, c in curves.items()})

    idxall = close.index
    hstart = idxall[idxall.get_indexer([pd.Timestamp(HIST_START)], method="bfill")[0]]
    sstart = idxall[idxall.get_indexer([pd.Timestamp(SIM_START)], method="bfill")[0]]
    days = list(idxall[idxall >= hstart])
    sim_start_idx = days.index(sstart)

    def select_on(d):
        best, bs = None, -1e9
        for n in agents:
            s = recency_weighted_sharpe(dret[n].loc[:d].dropna(), HALFLIFE)
            if s > bs: bs, best = s, n
        return best

    def target_on(agent_name, d):
        uni = screen_universe(close, d)
        return list(agents[agent_name].construct(close, rets, mkt, meta, d, uni).keys())

    def bench_on(agent_name, d, exclude, k=25):
        """선정 종목(target) 다음 순위 후보 — 거절/반려 시 대체용."""
        uni = screen_universe(close, d)
        try:
            r = agents[agent_name].rank(close, d, uni).dropna().sort_values(ascending=False)
        except Exception:
            return []
        ex = set(exclude)
        return [c for c in r.index if c not in ex][:k]

    plan = {}; cur = None
    for i, d in enumerate(days):
        dday = days[i-1] if i > 0 else idxall[idxall.get_loc(d)-1]
        if i % RESELECT_EVERY == 0:
            cur = select_on(dday)
        cash = krdata.riskoff_cash(close, dday)          # 하락장 방어 현금비율(0~1, 변동성 적응)
        tg = target_on(cur, dday)
        plan[d] = {"algo": cur, "target": tg,
                   "bench": bench_on(cur, dday, tg),     # 대체용 후순위 후보
                   "decision_day": dday, "reselect": (i % RESELECT_EVERY == 0),
                   "cash_ratio": cash}
    return dict(close=close, names=names, days=days, sim_start_idx=sim_start_idx, plan=plan,
                dret=dret, agents=agents, rets=rets, mkt=mkt, meta=meta)


def _ret(open_px, close_px, c):
    o = open_px.get(c); cl = close_px.get(c)
    if pd.isna(o) or pd.isna(cl) or float(o) <= 0:
        return 0.0
    return max(-0.35, min(0.35, float(cl)/float(o) - 1))   # ±30% 클립(불량가격 방어)


def _holdlist(names, open_px, close_px):
    held = S["held"]; per = S["invest"]/len(held) if held else 0
    out = []
    for c in held:
        v = per*(1+_ret(open_px, close_px, c)); o = open_px.get(c)
        qty = int(per/float(o)) if (not pd.isna(o) and float(o) > 0) else 0
        out.append({"code": c, "name": str(names.get(c,"")), "qty": qty, "value": int(v)})
    return sorted(out, key=lambda h: -h["value"])


def step_day(i, live):
    B = S["base"]; close = B["close"]; names = B["names"]
    day = B["days"][i]; pl = B["plan"][day]
    algo_auto = pl["algo"]
    algo = S.get("manual_algo") or algo_auto        # 수동 전환 시 그 전략으로 운용
    dday = pl["decision_day"]
    target = pl["target"]; bench = list(pl.get("bench", []))
    if algo != algo_auto:                           # 수동 전환 → 해당 전략의 목표 종목 재계산
        try:
            uni = screen_universe(close, dday)
            ag = B["agents"][algo]
            target = list(ag.construct(close, B["rets"], B["mkt"], B["meta"], dday, uni).keys())
            r = ag.rank(close, dday, uni).dropna().sort_values(ascending=False)
            bench = [c for c in r.index if c not in set(target)][:25]
        except Exception:
            algo = algo_auto; target = pl["target"]; bench = list(pl.get("bench", []))
    cash_ratio = pl.get("cash_ratio", 0.0)          # 하락장 방어 현금비율(0~1, 변동성 적응)
    open_px = close.loc[dday]; close_px = close.loc[day]
    events = []

    # 연 정산: 1월초(연 변경) → 300억 리셋
    if S.get("year") != day.year:
        if S.get("year") is not None:
            events.append({"type": "year", "msg": f"{S['year']}년 정산 완료 · {day.year}년 개시(한도 300억 리셋)"})
        S["year"] = day.year; S["limit"] = LIMIT_FULL; S["reduced"] = False
        S["equity"] = float(LIMIT_FULL); S["held"] = set(); S["invest"] = 0.0; S["cash"] = float(LIMIT_FULL)
        S["year_start"] = float(LIMIT_FULL); S["month_start"] = float(LIMIT_FULL); S["month"] = day.month
        S["freeze_until"] = -1; S["restore_idx"] = -1
        S["years"].setdefault(day.year, {"start": LIMIT_FULL, "curve": []})
        S["shadow"] = {n: float(LIMIT_FULL) for n in B["agents"]}   # 전략별 가상(paper) 평가액 리셋
    if S.get("month") != day.month:
        S["month"] = day.month; S["month_start"] = S["equity"]
    if S.get("reduced", False) and S["restore_idx"] > 0 and i >= S["restore_idx"]:
        S["limit"] = LIMIT_FULL; S["reduced"] = False
        events.append({"type": "restore", "msg": "월간 거래정지 해제 · 운용한도 정상 복구"})

    frozen = i <= S["freeze_until"]
    equity_start = S["equity"]                 # 시가 ≈ 전일 종가 → 전일 평가액
    sells, buys = [], []

    if not frozen:
        # 평가액 전액 배포(복리). 월간 손실로 한도 축소 시 축소 한도로 캡.
        # 하락장 방어: 현금비율만큼 투자 축소 (변동성 클수록 크게).
        base_cap = equity_start if S["limit"] >= LIMIT_FULL else S["limit"]
        deploy_cap = base_cap * (1 - cash_ratio)
        full_cash = cash_ratio >= 0.99
        tgt = ([c for c in target if not pd.isna(open_px.get(c)) and float(open_px.get(c)) > 0]
               if not full_cash else [])            # 전량 현금이면 목표 없음
        held = S["held"]
        entering = [c for c in tgt if c not in held]
        benchq = [c for c in bench if c not in held and c not in tgt
                  and not pd.isna(open_px.get(c)) and float(open_px.get(c) or 0) > 0]
        inject = {}
        if live and entering and i % 3 == 1: inject[entering[-1]] = "vi"
        if live and len(entering) > 1 and i % 4 == 2: inject[entering[0]] = "halted"
        n_slots = max(len(tgt), 1)
        per_budget = int(deploy_cap / n_slots)

        def mo_check(code):
            raw = float(open_px[code]); t = kr_tick(raw); px = int(round(raw/t)*t)
            o = dict(symbol=code, name=str(names.get(code,"")), side="BUY", price=px, qty=1,
                     order_value=per_budget, prev_close=raw, last_price=px, tick_size=t,
                     halted=(inject.get(code)=="halted"), vi_active=(inject.get(code)=="vi"),
                     adv=krdata.adv(dday, code))
            res = check_compliance(o)
            rej = "" if res.passed else "; ".join(f"{v.reg_id} {v.title}" for v in res.violations)
            flg = "; ".join(f"{fl.reg_id} {fl.title}" for fl in res.flags)
            return px, res.passed, rej, flg

        approved_new, buys_rec = [], []
        used = set(tgt) | set(held)
        queue = list(entering)
        while queue:
            c = queue.pop(0)
            px, passed, rej, flg = mo_check(c)
            if passed:
                approved_new.append(c)
                buys_rec.append((c, str(names.get(c,"")), px, True, "", flg, None))
            else:
                # M/O 반려 → 체결 안 함. 다음 순위 후보로 대체(연쇄).
                sub = None
                while benchq:
                    cand = benchq.pop(0)
                    if cand in used: continue
                    used.add(cand); sub = cand; break
                buys_rec.append((c, str(names.get(c,"")), px, False, rej, flg,
                                 str(names.get(sub,"")) if sub else None))  # 반려(대체 안내)
                if sub:
                    queue.append(sub)                # 대체 종목도 M/O 심사(또 반려면 다음으로)
        final = [c for c in tgt if c in held] + approved_new
        for c in held:
            if c not in final:
                px = open_px.get(c)
                q = int((S["invest"]/max(len(held),1))/float(px)) if (not pd.isna(px) and float(px)>0) else 0
                amt = int(q*float(px)) if not pd.isna(px) else 0
                sells.append({"code": c, "name": str(names.get(c,"")), "qty": q,
                    "price": int(px) if not pd.isna(px) else 0,
                    "amount": amt, "why": ("하락장" if full_cash else ""),
                    "need_human": (amt >= HUMAN_VALUE) if not full_cash else False})
        S["held"] = set(final)
        S["invest"] = min(equity_start, deploy_cap); S["cash"] = equity_start - S["invest"]
        per = S["invest"]/len(final) if final else 0
        for c, nm, px, passed, reject_reason, flag_reason, sub_name in buys_rec:
            qty = int(per/float(open_px[c])) if (passed and float(open_px[c]) > 0) else 0
            amount = int(qty*px)
            if not passed:                                        # M/O 반려 → 체결 안 됨(대체됨)
                verdict = "REJECTED"
                reason = reject_reason + (f" → 대체: {sub_name}" if sub_name else " → 대체 후보 없음")
                amount = 0
            elif amount >= HUMAN_VALUE or flag_reason:            # 30억↑ 또는 주의 → 딜러 결재
                verdict = "NEEDS_HUMAN"
                reason = flag_reason or f"주문금액 {amount/1e8:.0f}억 ≥ 30억 결재선"
            else:                                                 # 통과 + 한도 미만 → 시가 자동체결
                verdict, reason = "AUTO", ""
            buys.append({"code": c, "name": nm, "qty": qty, "price": px,
                         "amount": amount, "verdict": verdict, "reason": reason})
        # 딜러 거절 시 대체용: 그날 컨텍스트 저장
        S["today"] = {"dday": dday.isoformat(), "day_i": i, "algo_kr": AGENT_KR.get(algo, algo),
                      "bench": [c for c in benchq if c not in S["held"]], "per": per}

    # 종가 정산 (수익률 기반)
    port_ret = (sum(_ret(open_px, close_px, c) for c in S["held"]) / len(S["held"])) if S["held"] else 0.0
    equity_end = S["cash"] + S["invest"]*(1+port_ret)
    day_pnl = equity_end - equity_start
    S["equity"] = equity_end
    S["years"][day.year]["curve"].append((day.date().isoformat(), equity_end))

    if not frozen:
        daily_limit = -DAILY_LOSS_PCT * equity_start            # 그날 시작 평가액의 10%
        if day_pnl <= daily_limit:
            S["freeze_until"] = i + DAILY_FREEZE
            events.append({"type": "daily", "msg":
                f"⚠ 일일 손실한도 도달 (당일 {day_pnl/1e8:.0f}억 / 한도 {daily_limit/1e8:.0f}억 = 평가액의 10%) "
                f"— {DAILY_FREEZE}영업일 신규매매 정지"})
        month_pnl = equity_end - S["month_start"]
        monthly_limit = -MONTHLY_LOSS_PCT * S["month_start"]    # 월초 평가액의 30%
        if month_pnl <= monthly_limit and not S.get("reduced", False):
            reduced_cap = equity_end * REDUCE_FACTOR            # 운용한도 = 평가액의 1/3
            S["limit"] = reduced_cap; S["reduced"] = True
            order = sorted(S["held"], key=lambda c: _ret(open_px, close_px, c))  # 손실 큰 종목 우선
            drop = order[:len(order)//2]
            for c in drop:
                sells.append({"code": c, "name": str(names.get(c,"")), "qty": 0, "amount": 0,
                              "price": int(close_px.get(c)) if not pd.isna(close_px.get(c)) else 0,
                              "why": "한도축소", "need_human": False})
            S["held"] = set(order[len(order)//2:])
            S["invest"] = min(equity_end, reduced_cap); S["cash"] = equity_end - S["invest"]
            S["freeze_until"] = i + MONTHLY_FREEZE; S["restore_idx"] = i + MONTHLY_FREEZE
            events.append({"type": "monthly", "msg":
                f"🚨 [딜러 통보] 월간 손실한도 도달 (당월 {month_pnl/1e8:.0f}억 / 한도 {monthly_limit/1e8:.0f}억 = 월초의 30%) "
                f"— 운용한도 {reduced_cap/1e8:.0f}억(평가액 1/3)로 축소, 손실 종목 매도, {MONTHLY_FREEZE}영업일 신규매매 정지"})

    # 하락장 방어 발동/해제 알람 (현금비율 상태 전환 시)
    prev_def = S.get("defend_on", False)
    now_def = (cash_ratio >= 0.15) and not frozen
    if now_def and not prev_def:
        events.append({"type": "defend", "msg":
            f"🛡️ 하락장 방어 발동 — 변동성 급증·추세 이탈, 현금 비중 {cash_ratio*100:.0f}% 확보"})
    elif prev_def and not now_def and not frozen:
        events.append({"type": "restore", "msg": "🌤️ 시장 안정 — 하락장 방어 해제, 정상 운용 복귀"})
    S["defend_on"] = now_def

    S["hold_list"] = _holdlist(names, open_px, close_px)
    if live:                                   # 진행 거래에 유니크 거래번호 + 매매이유 부여
        ds = day.date().isoformat(); akr = AGENT_KR.get(algo, algo)
        for s in sells:
            kind = "축소" if s.get("why") == "한도축소" else ("하락장" if s.get("why") == "하락장" else "매도")
            s["tid"] = new_trade_id(ds, s["code"], s["name"], "SELL", akr,
                                    s["price"], s["qty"], kind, dday)
            s["trade_reason"] = S["trades"][s["tid"]]["reason"]     # 저장된 매도 이유
        for b in buys:
            if b["verdict"] in ("AUTO", "NEEDS_HUMAN"):   # 체결 대상(자동/딜러승인)에 거래번호 부여
                b["tid"] = new_trade_id(ds, b["code"], b["name"], "BUY", akr,
                                        b["price"], b["qty"], "매수", dday)
                b["trade_reason"] = S["trades"][b["tid"]]["reason"]  # 저장된 매수 이유
    year_ret = (equity_end - S["year_start"]) / S["year_start"] * 100

    # 전략별 가상(paper) 손익 갱신 — 각 전략의 순수 백테스트 일수익으로 복리
    dret = B["dret"]
    S.setdefault("shadow", {n: float(LIMIT_FULL) for n in B["agents"]})
    for n in list(S["shadow"]):
        r = dret.at[day, n] if (day in dret.index and n in dret.columns) else 0.0
        if pd.isna(r):
            r = 0.0
        S["shadow"][n] *= (1 + float(r))
    shadow = {n: round((S["shadow"][n] / S["year_start"] - 1) * 100, 2) for n in S["shadow"]}

    return {"date": day.date().isoformat(), "year": day.year,
            "algo": algo, "algo_kr": AGENT_KR.get(algo, algo), "algo_auto": algo_auto,
            "manual": bool(S.get("manual_algo")), "reselect": pl["reselect"],
            "frozen": frozen, "cash_ratio": round(cash_ratio, 3),
            "defend": cash_ratio >= 0.15, "limit": int(S["limit"]),
            "sells": sells, "buys": buys, "holdings": S["hold_list"],
            "day_pnl": int(day_pnl), "month_pnl": int(equity_end - S["month_start"]),
            "equity": int(equity_end), "year_return": round(year_ret, 2),
            "shadow": shadow, "events": events}


def reset_state():
    B = S["base"]
    S.update(equity=float(LIMIT_FULL), held=set(), invest=0.0, cash=float(LIMIT_FULL),
             year=None, month=None, limit=LIMIT_FULL, reduced=False, freeze_until=-1, restore_idx=-1,
             years={}, hold_list=[], trades={}, trade_seq=0,
             manual_algo=None, shadow={n: float(LIMIT_FULL) for n in B["agents"]})
    for i in range(B["sim_start_idx"]):
        step_day(i, live=False)
    S["idx"] = B["sim_start_idx"] - 1


def new_trade_id(date, code, name, side, algo_kr, price, qty, kind, dday):
    """유니크 거래번호 발급 + 맥락·매매이유 저장(팝업 표시 및 소명 질의용)."""
    S["trade_seq"] += 1
    tid = f"T-{date.replace('-','')}-{S['trade_seq']:04d}"
    ctx = _stock_context(code, dday)
    reason = _rule_reason(side, algo_kr, ctx, kind)   # 즉시 저장(규칙). LLM 설명은 조회 시 갱신
    S["trades"][tid] = {"id": tid, "date": date, "code": code, "name": name, "side": side,
                        "kind": kind, "algo_kr": algo_kr, "price": price, "qty": qty,
                        "ctx": ctx, "reason": reason, "reason_by": "규칙"}
    return tid


def year_summary():
    out = []
    for y in sorted(S["years"]):
        cur = S["years"][y]["curve"]
        if not cur: continue
        start = S["years"][y]["start"]; end = cur[-1][1]
        step = max(1, len(cur)//80)
        out.append({"year": y, "return_pct": round((end-start)/start*100, 2),
                    "end_equity": int(end),
                    "curve": [{"d": d, "e": round(e/start, 4)} for d, e in cur[::step]]})
    return out


@app.route("/")
def index():
    return send_file(os.path.join(HERE, "sim_desk.html"))


@app.route("/api/sim/init")
def sim_init():
    B = S["base"]
    shadow = {n: round((S["shadow"][n] / float(LIMIT_FULL) - 1) * 100, 2) for n in S["shadow"]} \
        if S.get("shadow") else {}
    return jsonify({
        "years": year_summary(), "current_year": B["days"][B["sim_start_idx"]-1].year,
        "limit": int(S["limit"]), "equity": int(S["equity"]),
        "holdings": S["hold_list"],
        "agents": [{"name": n, "kr": AGENT_KR.get(n, n)} for n in B["agents"]],
        "shadow": shadow, "manual": S.get("manual_algo"),
        "sim_start": B["days"][B["sim_start_idx"]].date().isoformat(),
        "sim_end": B["days"][-1].date().isoformat(),
        "sim_total": len(B["days"]) - B["sim_start_idx"],
    })


@app.route("/api/sim/set_algo", methods=["POST"])
def sim_set_algo():
    """봇 클릭 → 그 전략으로 수동 전환. algo=None/미존재면 자동 선정으로 복귀."""
    a = request.get_json(force=True).get("algo")
    S["manual_algo"] = a if a in S["base"]["agents"] else None
    return jsonify({"ok": True, "manual": S["manual_algo"]})


@app.route("/api/sim/next", methods=["POST"])
def sim_next():
    B = S["base"]
    if S["idx"] >= len(B["days"]) - 1:
        return jsonify({"done": True})
    S["idx"] += 1
    r = step_day(S["idx"], live=True); r["done"] = False
    r["day_no"] = S["idx"] - B["sim_start_idx"] + 1
    r["sim_total"] = len(B["days"]) - B["sim_start_idx"]
    return jsonify(r)


@app.route("/api/sim/reset", methods=["POST"])
def sim_reset():
    reset_state(); return jsonify({"ok": True})


def _stock_context(code, dday):
    """종목의 매매 시점 상황(추세·과매도 등) 지표 — 설명 근거용."""
    close = S["base"]["close"]
    s = close[code].loc[:dday].dropna()
    if len(s) < 65:
        return {}
    p = float(s.iloc[-1])
    hi60 = float(s.iloc[-60:].max())
    ctx = {"ret_5": s.iloc[-1]/s.iloc[-6]-1, "ret_20": s.iloc[-1]/s.iloc[-21]-1,
           "ret_60": s.iloc[-1]/s.iloc[-61]-1, "gap_high60": p/hi60-1,
           "near_high": p >= hi60*0.98}
    return ctx


def _rule_reason(side, algo_kr, ctx, kind=""):
    r20 = ctx.get("ret_20", 0) * 100
    r60 = ctx.get("ret_60", 0) * 100
    if side == "BUY":
        base = {"모멘텀": "최근 상승 모멘텀이 강해 편입",
                "돌파": "신고가 돌파로 추세 진입",
                "평균회귀": "과매도 구간에서 반등을 기대해 편입",
                "시총상위10": "시가총액 상위 대형주로 편입",
                "저변동성": "변동성이 낮아 방어형 포트폴리오에 편입",
                "52주신고가": "52주 고점에 근접해 상승 지속을 기대하고 편입",
                "거래량": "가격과 거래량이 동반 상승해 매수세 유입으로 판단",
                "BuyHold순수": "시총 상위 바스켓으로 장기 보유 목적 매수",
                "BuyHold피라미딩": "수익 구간이라 추가 매수(피라미딩)",
                }.get(algo_kr, "선정 전략의 목표 종목이라 편입")
        near = ", 신고가 부근" if ctx.get("near_high") else ""
        return f"{base} (최근 20일 {r20:+.0f}%, 60일 {r60:+.0f}%{near})."
    if kind == "축소":
        return f"월간 손실한도 도달 → 운용한도 축소에 따른 강제 매도 (최근 20일 {r20:+.0f}%)."
    if kind == "하락장":
        return f"시장 국면이 하락으로 전환돼 리스크 축소 매도 (최근 20일 {r20:+.0f}%)."
    return f"목표 포트폴리오에서 제외돼 매도 — 추세 약화·전략 조건 이탈 (최근 20일 {r20:+.0f}%)."


def explain_trade(code, name, side, algo_kr, ctx, kind=""):
    """LLM(ChatGPT 4o)이 매수/매도 근거를 한국어 1~2문장으로 설명. 실패 시 규칙 문장."""
    if not llm.available() or not ctx:
        return _rule_reason(side, algo_kr, ctx, kind)
    system = ("너는 한국 증권사 딜러다. 선정된 전략과 종목의 최근 흐름을 근거로 "
              "왜 이 종목을 사는지/파는지 한국어 1~2문장으로 간결히 설명한다. 과장 없이 수치를 근거로.")
    act = "매수" if side == "BUY" else "매도"
    extra = {"축소": " (월간 손실한도로 운용한도가 축소돼 강제 매도)",
             "하락장": " (시장 국면이 하락으로 전환돼 리스크 축소)"}.get(kind, "")
    user = (f"전략: {algo_kr}\n종목: {name}({code})\n행동: {act}{extra}\n"
            f"최근 5일 {ctx['ret_5']*100:+.1f}%, 20일 {ctx['ret_20']*100:+.1f}%, "
            f"60일 {ctx['ret_60']*100:+.1f}%, 60일고점대비 {ctx['gap_high60']*100:+.1f}%"
            f"{', 신고가 부근' if ctx.get('near_high') else ''}")
    return llm.chat(system, user, max_tokens=140) or _rule_reason(side, algo_kr, ctx, kind)


@app.route("/api/sim/explain", methods=["POST"])
def sim_explain():
    """매수/매도 근거 설명. 거래번호(tid)가 오면 그 거래에 이유를 저장(소명 질의용)."""
    B = S["base"]; req = request.get_json(force=True)
    tid = req.get("tid")
    tr = S.get("trades", {}).get(tid) if tid else None

    if tr:                                   # 저장된 거래 → 그 시점 맥락으로 설명
        code, name, side, algo_kr = tr["code"], tr["name"], tr["side"], tr["algo_kr"]
        ctx, kind = tr.get("ctx") or {}, tr.get("kind", "")
    else:                                    # tid 없으면 현재 진행일 기준
        code = req.get("code"); side = req.get("side", "BUY"); name = req.get("name", "")
        i = max(S["idx"], B["sim_start_idx"])
        day = B["days"][min(i, len(B["days"])-1)]
        dday = B["plan"][day]["decision_day"]
        algo_kr = AGENT_KR.get(B["plan"][day]["algo"], B["plan"][day]["algo"])
        ctx, kind = _stock_context(code, dday), req.get("kind", "")

    reason = explain_trade(code, name, side, algo_kr, ctx, kind)
    by = "LLM(ChatGPT 4o)" if (llm.available() and ctx) else "규칙"
    if tr:                                   # 생성된 설명을 거래에 되저장 → 소명 질의에서 재사용
        tr["reason"] = reason; tr["reason_by"] = by
    return jsonify({"reason": reason, "by": by})


@app.route("/api/sim/dealer_reject", methods=["POST"])
def sim_dealer_reject():
    """딜러가 재량으로 거래를 거절 → 체결 취소(보유·거래번호 제거) 후 다음 순위 종목으로 대체."""
    tid = request.get_json(force=True).get("tid")
    tr = S.get("trades", {}).get(tid)
    if not tr:
        return jsonify({"ok": False, "msg": "거래를 찾을 수 없습니다."})
    B = S["base"]; close = B["close"]; names = B["names"]
    rejected_code = tr["code"]
    # 1) 체결 취소: 보유에서 제거 + 거래번호 폐기
    S["held"].discard(rejected_code)
    S["trades"].pop(tid, None)

    td = S.get("today") or {}
    dday = pd.Timestamp(td.get("dday")) if td.get("dday") else None
    benchq = [c for c in td.get("bench", []) if c not in S["held"]]
    per = td.get("per", 0); akr = td.get("algo_kr", tr.get("algo_kr",""))
    ds = tr["date"]

    # 2) 다음 순위 후보로 대체 (M/O 통과할 때까지 연쇄)
    sub = None
    if dday is not None:
        open_px = close.loc[dday]
        while benchq:
            cand = benchq.pop(0)
            raw = open_px.get(cand)
            if pd.isna(raw) or float(raw or 0) <= 0:
                continue
            t = kr_tick(float(raw)); px = int(round(float(raw)/t)*t)
            o = dict(symbol=cand, name=str(names.get(cand,"")), side="BUY", price=px, qty=1,
                     order_value=int(per) if per else 0, prev_close=float(raw), last_price=px,
                     tick_size=t, halted=False, vi_active=False, adv=krdata.adv(dday, cand))
            if not check_compliance(o).passed:
                continue                                  # 대체 후보도 반려면 다음으로
            qty = int(per/float(raw)) if float(raw) > 0 else 0
            ntid = new_trade_id(ds, cand, str(names.get(cand,"")), "BUY", akr, px, qty, "매수", dday)
            S["held"].add(cand)
            sub = {"tid": ntid, "code": cand, "name": str(names.get(cand,"")),
                   "price": px, "qty": qty, "amount": int(qty*px),
                   "reason": S["trades"][ntid]["reason"]}
            break
    S["today"]["bench"] = benchq
    return jsonify({"ok": True, "rejected": {"tid": tid, "name": tr["name"]}, "substitute": sub})


@app.route("/api/sim/ask", methods=["POST"])
def sim_ask():
    """거래번호로 '왜 이 거래를 했는지' 소명 질의 → LLM이 저장된 맥락으로 답변."""
    req = request.get_json(force=True)
    tid = req.get("tid", "").strip().upper(); q = req.get("question", "").strip()
    tr = S.get("trades", {}).get(tid)
    if not tr:
        return jsonify({"answer": f"거래번호 {tid} 를 찾을 수 없습니다.", "by": "-", "found": False})
    saved = tr.get("reason", "")          # 거래 시점에 저장된 매매 이유
    ctx = tr.get("ctx") or {}
    ctxline = (f"5일 {ctx.get('ret_5',0)*100:+.1f}%, 20일 {ctx.get('ret_20',0)*100:+.1f}%, "
               f"60일 {ctx.get('ret_60',0)*100:+.1f}%, 60일고점대비 {ctx.get('gap_high60',0)*100:+.1f}%"
               f"{', 신고가 부근' if ctx.get('near_high') else ''}") if ctx else "지표 없음"
    act = "매수" if tr["side"] == "BUY" else "매도"
    reject_note = " (딜러가 최종 거절함)" if tr.get("dealer_rejected") else ""
    if llm.available():
        system = ("너는 한국 증권사 딜러다. 과거 거래에 대한 감사·소명 질의에 답한다. "
                  "저장된 거래 맥락(선정 전략·종목 흐름)을 근거로, 왜 그 거래를 했는지 한국어로 간결히 설명한다. "
                  "질문이 있으면 그 질문에 초점을 맞춰 답한다.")
        user = (f"[거래] {tid} · {tr['date']} · {tr['name']}({tr['code']}) {act} "
                f"{tr['qty']:,}주 @ {tr['price']:,}원{reject_note}\n"
                f"[선정 전략] {tr['algo_kr']}\n[당시 종목 흐름] {ctxline}\n"
                f"[거래 시점 기록된 사유] {saved}\n\n"
                f"[질문] {q or '이 거래를 한 이유가 무엇인가?'}")
        ans = llm.chat(system, user, max_tokens=260)
        if ans:
            return jsonify({"answer": ans, "by": "LLM(ChatGPT 4o)", "found": True, "trade": {
                "id": tid, "date": tr["date"], "name": tr["name"], "side": act,
                "algo_kr": tr["algo_kr"], "qty": tr["qty"], "price": tr["price"],
                "dealer_rejected": tr.get("dealer_rejected", False)}})
    # 규칙 폴백
    ans = (f"{tr['date']} {tr['algo_kr']} 전략 선정에 따라 {tr['name']}를 {act}했습니다.\n"
           f"사유: {saved}\n당시 흐름: {ctxline}.{reject_note}")
    return jsonify({"answer": ans, "by": "규칙", "found": True, "trade": {
        "id": tid, "date": tr["date"], "name": tr["name"], "side": act,
        "algo_kr": tr["algo_kr"], "qty": tr["qty"], "price": tr["price"],
        "dealer_rejected": tr.get("dealer_rejected", False)}})


def try_update_prices():
    """서버 시작 시 가장 최근 날짜 이후 시세를 증분 갱신. 갱신 불가 환경이면 기존 데이터로 진행.
    끄려면 환경변수 DESK_AUTOUPDATE=0."""
    if os.environ.get("DESK_AUTOUPDATE", "1") != "1":
        print("[시세 갱신 건너뜀] DESK_AUTOUPDATE=0")
        return
    try:
        import update_prices
        print("시세 갱신 중… (가장 최근 날짜 이후 증분 수집)")
        update_prices.main()
    except Exception as e:
        print(f"[시세 갱신 생략] {e} — 기존 kr_prices.csv 로 진행")


if __name__ == "__main__":
    try_update_prices()                      # ← 최신 날짜 이후 시장 데이터 업데이트
    print("사전 계산 중… (전 구간 월1회 선정·목표종목)")
    S["base"] = build_sim()
    reset_state()
    for y in year_summary():
        print(f"  {y['year']}년 누적손익 {y['return_pct']:+.1f}% (평가액 {y['end_equity']:,})")
    print(f"6월 시뮬 {S['base']['days'][S['base']['sim_start_idx']].date()} ~ {S['base']['days'][-1].date()}")
    print("서버 시작 → http://localhost:5001")
    app.run(host="0.0.0.0", port=5001, debug=False)