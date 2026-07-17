"""
통합 서버 — HTML 데스크 화면 + 실제 파이썬 파이프라인을 함께 구동

startup: 실데이터로 파이프라인 실행(반감기 비교 선정 → 우승 룰 포트폴리오 주문 → M/O 검사)
routes:
  GET  /                → 데스크 HTML
  GET  /api/pipeline    → 선정 결과 + 주문 + M/O 판정 (실데이터로 계산된 값)
  POST /api/decision    → 결재(승인/거부) 기록 및 체결 확정
  POST /api/reset       → 감사 로그 초기화

실행:  pip install flask  후   python app.py   → http://localhost:5000
필요 모듈: krdata, kr_agents, kr_backtest, mo_engine (+ 데이터 CSV)
"""
import os
import numpy as np
from flask import Flask, jsonify, request, send_file

import krdata
from krdata import load_data, daily_returns, market_return, screen_universe
from kr_agents import make_agents
from kr_backtest import select_with_halflife
from mo_engine import check_compliance, HUMAN_VALUE
import llm   # OpenAI(ChatGPT 4o); 키 없으면 규칙 기반 폴백

HERE = os.path.dirname(os.path.abspath(__file__))
CAPITAL = 30_000_000_000                     # 운용 300억 (데스크 한도) — 상위 컨빅션 주문이 30억 결재선을 넘도록
MARKET_FLAGS = {}                            # 아래에서 우승 종목 중 2개에 거래정지/VI 주입

AGENT_KR = {"momentum": "모멘텀", "mean_reversion": "평균회귀",
            "breakout": "돌파", "top_cap": "시총상위10",
            "low_vol": "저변동성", "week52_high": "52주신고가", "volume_flow": "거래량"}

app = Flask(__name__)
_CACHE = {"pipeline": None, "audit": []}


def kr_tick(p):
    for lim, t in [(2000, 1), (5000, 5), (20000, 10), (50000, 50),
                   (200000, 100), (500000, 500)]:
        if p < lim:
            return t
    return 1000


def build_pipeline():
    close, meta = load_data()
    rets = daily_returns(close)
    mkt = market_return(rets)

    # 1) 반감기 비교 선정 (실데이터)
    results, best_hl = select_with_halflife(close, rets, mkt, meta, train_frac=0.6)
    selection = [{"halflife": L, "winner": r["winner"],
                  "winner_kr": AGENT_KR[r["winner"]],
                  "wsharpe": round(float(r["wsharpe"]), 2),
                  "chosen": (L == best_hl)}
                 for L, r in results.items()]
    win = results[best_hl]
    H = win["H"]
    winner = win["winner"]

    # 선정 근거(말로 설명) + 후보 비교(그래프용)
    rows = win["rows"]
    valid_best = max(rows, key=lambda k: rows[k]["valid"]["sharpe"])
    robust = (winner == valid_best)
    tw = float(win["wsharpe"])
    vs = float(rows[winner]["valid"]["sharpe"])
    vr = float(rows[winner]["valid"]["total"])
    rule_rationale = (
        f"{AGENT_KR[winner]}(반감기 {best_hl}) 채택 — "
        f"학습구간 가중샤프 {tw:.2f}로 후보 중 1위. "
        f"검증구간 샤프 {vs:.2f}·수익 {vr*100:+.0f}%. "
        + ("검증구간에서도 1위라 견고합니다." if robust
           else f"다만 검증 1위는 {AGENT_KR[valid_best]}로, 국면 전환 주의가 필요합니다.")
    )
    comparison = [{"agent_kr": AGENT_KR[n],
                   "valid_sharpe": round(float(r["valid"]["sharpe"]), 2),
                   "valid_return": round(float(r["valid"]["total"]) * 100, 1),
                   "chosen": (n == winner)}
                  for n, r in sorted(rows.items(), key=lambda x: -x[1]["valid"]["sharpe"])]

    # 선정 근거 — LLM(ChatGPT 4o)이 시니어 딜러 브리핑으로 설명 (실패 시 규칙 문장)
    rationale = llm_selection_rationale(AGENT_KR[winner], best_hl, comparison,
                                        AGENT_KR[valid_best], robust) or rule_rationale
    rationale_by = "LLM(ChatGPT 4o)" if llm.available() else "규칙"

    # 2) 우승 룰의 포트폴리오 → 주문 (컨빅션 가중으로 규모 다양화)
    agent = {a.name: a for a in make_agents(H)}[winner]
    date = close.index[-1]
    uni = screen_universe(close, date)
    weights = agent.construct(close, rets, mkt, meta, date, uni)
    scores = agent.rank(close, date, uni).reindex(weights.keys()).clip(lower=1e-6)
    scores = scores / scores.sum()
    names = meta.set_index("code")["name"]

    picks = list(scores.sort_values(ascending=False).index)[:10]
    # 실시간 시장상태 가정: 일부 종목에 거래정지 / VI 주입 (데모) — 통과분은 남김
    flags = {}
    if len(picks) >= 5:
        flags[picks[-1]] = {"halted": True}
        flags[picks[-2]] = {"vi": True}
    elif len(picks) >= 3:
        flags[picks[-1]] = {"vi": True}

    open_panel = krdata.panel("open")            # 시가 패널 (없으면 종가로 폴백)
    orders = []
    for code in picks:
        raw = float(close.at[date, code])
        t = kr_tick(raw)
        price = int(round(raw / t) * t)          # 주문 기준가(전일 종가)
        value = int(CAPITAL * float(scores[code]))
        qty = max(1, value // price)
        # 시가 체결가: 실행일 시가를 호가단위로 반올림 (시가 없으면 종가 폴백)
        raw_open = float(open_panel.at[date, code]) if open_panel is not None else raw
        if np.isnan(raw_open):
            raw_open = raw
        ft = kr_tick(raw_open)
        fill_price = int(round(raw_open / ft) * ft)
        f = flags.get(code, {})
        o = dict(symbol=code, name=str(names.get(code, "")), side="BUY",
                 price=price, qty=qty, order_value=qty * price,
                 prev_close=float(close.iloc[-2][code]), last_price=price,
                 tick_size=t, halted=f.get("halted", False), vi_active=f.get("vi", False),
                 adv=krdata.adv(date, code))       # ADV → REG-010 시장충격 검사
        res = check_compliance(o, use_llm=True)   # LLM(ChatGPT 4o) 규정 판단 부착
        if not res.passed:                                        # M/O 위반 → 반려
            verdict = "REJECTED"
            reasons = [v.reg_id + " " + v.title for v in res.violations]
        elif o["order_value"] >= HUMAN_VALUE or res.flags:        # 30억↑ 또는 주의 플래그 → 딜러 승인
            verdict = "NEEDS_HUMAN"
            reasons = ([fl.reg_id + " " + fl.title for fl in res.flags]
                       or [f"주문금액 {o['order_value']/1e8:.0f}억 ≥ 30억 결재선"])
        else:                                                     # 통과 + 한도 미만 → 자동승인(시가 체결)
            verdict = "AUTO"
            reasons = []
        orders.append({"code": code, "name": o["name"], "qty": qty,
                       "price": price, "fill_price": fill_price,
                       "value": o["order_value"], "fill_value": qty * fill_price,
                       "verdict": verdict, "reasons": reasons,
                       "llm": res.llm})          # LLM 판단 {verdict, reason, regs} 또는 None

    return {"selection": selection,
            "chosen": {"halflife": best_hl, "agent": winner, "agent_kr": AGENT_KR[winner]},
            "rationale": rationale, "rationale_by": rationale_by,
            "llm_on": llm.available(),
            "comparison": comparison,
            "orders": orders,
            "period": f"{close.index.min().date()} ~ {close.index.max().date()}"}


def llm_selection_rationale(winner_kr, best_hl, comparison, valid_best_kr, robust):
    """시니어 딜러 브리핑 — 전략 경쟁 결과를 LLM이 자연어로 설명. 실패 시 None."""
    if not llm.available():
        return None
    lines = "\n".join(
        f"- {c['agent_kr']}: 검증샤프 {c['valid_sharpe']}, 검증수익 {c['valid_return']}%"
        + ("  ← 채택" if c["chosen"] else "") for c in comparison)
    system = ("너는 한국 증권사의 시니어 딜러다. 아래 전략 경쟁(백테스트) 결과를 근거로 "
              "왜 이 전략을 선택했는지 팀에게 브리핑하듯 3문장 이내 한국어로 설명한다. "
              "위험조정성과(샤프)를 중심으로 설명하고, 과장 없이. "
              "검증 1위와 채택이 다르면 국면 전환 위험을 반드시 언급한다.")
    user = (f"채택: {winner_kr} (반감기 {best_hl})\n검증 샤프 1위: {valid_best_kr}\n"
            f"견고성: {'검증에서도 1위' if robust else '검증 1위와 다름'}\n\n후보 성과:\n{lines}")
    return llm.chat(system, user, max_tokens=260)


@app.route("/")
def index():
    return send_file(os.path.join(HERE, "trading_desk.html"))


@app.route("/api/pipeline")
def api_pipeline():
    if _CACHE["pipeline"] is None:
        _CACHE["pipeline"] = build_pipeline()
    _CACHE["audit"] = []
    return jsonify(_CACHE["pipeline"])


@app.route("/api/decision", methods=["POST"])
def api_decision():
    d = request.get_json(force=True)
    _CACHE["audit"].append(d)
    filled = d.get("decision") in ("APPROVED", "AUTO_FILLED")
    return jsonify({"ok": True, "code": d.get("code"), "filled": filled})


@app.route("/api/reset", methods=["POST"])
def api_reset():
    _CACHE["audit"] = []
    return jsonify({"ok": True})


def try_update_prices():
    """서버 시작 시 오늘자 시세 갱신 시도. 갱신 불가 환경이면 기존 데이터로 진행."""
    if os.environ.get("DESK_AUTOUPDATE", "1") != "1":
        return
    try:
        import update_prices
        update_prices.main()
    except Exception as e:
        print(f"[시세 갱신 생략] {e} — 기존 kr_prices.csv 로 진행")


if __name__ == "__main__":
    try_update_prices()
    print("파이프라인 계산 중… (실데이터 반감기 비교 선정)")
    _CACHE["pipeline"] = build_pipeline()
    ch = _CACHE["pipeline"]["chosen"]
    print(f"채택: 반감기 {ch['halflife']} · {ch['agent_kr']} | 주문 {len(_CACHE['pipeline']['orders'])}건")
    print("서버 시작 → http://localhost:5002")
    app.run(host="0.0.0.0", port=5002, debug=False)