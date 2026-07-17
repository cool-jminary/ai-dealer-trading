"""
최종 오케스트레이션 (한국 실데이터) — 시니어 딜러(Supervisor)가 전 과정을 지휘

흐름 (LangGraph / Supervisor 패턴):
  START → select(워크포워드 선정)
        → build_orders(우승 룰의 포트폴리오 → 매수 주문들)
        → mo_review(각 주문을 KRX 규정 M/O로 검사: 승인/반려/사람결재)
        → [사람결재 필요?] → human_gate(HITL interrupt) → execute
                          → execute(승인분 체결) → END

재사용: krdata, kr_agents, kr_backtest(선정), mo_engine(RAG+규정검사)
"""
import operator
from typing import TypedDict, Annotated

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command

import krdata
from krdata import load_data, daily_returns, market_return, screen_universe
from kr_agents import make_agents
from kr_backtest import walk_forward_select
from mo_engine import search_documents, check_compliance, HUMAN_VALUE

CAPITAL = 30_000_000_000     # 운용 자본 300억 (기관 데스크 규모)
# 오늘의 시장 상태 플래그 (실무: 실시간 KRX 피드) — 데모용 가정
MARKET_FLAGS = {"007810": {"halted": True},   # 코리아써키트: 거래정지 가정
                "000990": {"vi": True}}       # DB하이텍: VI 발동 가정


# 데이터는 State(체크포인터 직렬화) 밖에 둔다
DATA = {}


def kr_tick(p):
    if p < 2000: return 1
    if p < 5000: return 5
    if p < 20000: return 10
    if p < 50000: return 50
    if p < 200000: return 100
    if p < 500000: return 500
    return 1000


def round_tick(p):
    t = kr_tick(p)
    return int(round(p / t) * t)


# ======================================================================
# State
# ======================================================================
class OrchState(TypedDict):
    selected_agent: str
    selection_report: dict
    orders: list
    blotter: list
    needs_human: bool
    decision: str
    logs: Annotated[list, operator.add]


# ======================================================================
# 노드
# ======================================================================
def select_node(state):
    close, rets, mkt, meta = DATA["close"], DATA["rets"], DATA["mkt"], DATA["meta"]
    _, rows, winner, valid_best, split = walk_forward_select(
        make_agents(), close, rets, mkt, meta, train_frac=0.6)
    report = {n: {"train_sharpe": round(float(r["train"]["sharpe"]), 2),
                  "valid_sharpe": round(float(r["valid"]["sharpe"]), 2)}
              for n, r in rows.items()}
    robust = "견고" if winner == valid_best else "과최적화 의심"
    return {"selected_agent": winner, "selection_report": report,
            "logs": [f"[select] 학습 1등={winner} (검증 1등={valid_best} · {robust})"]}


def build_orders_node(state):
    close, rets, mkt, meta = DATA["close"], DATA["rets"], DATA["mkt"], DATA["meta"]
    names = meta.set_index("code")["name"]
    agent = {a.name: a for a in make_agents()}[state["selected_agent"]]
    date = close.index[-1]
    uni = screen_universe(close, date)
    weights = agent.construct(close, rets, mkt, meta, date, uni)

    orders = []
    for code, w in sorted(weights.items(), key=lambda x: -x[1]):
        price = round_tick(float(close.at[date, code]))
        qty = int(w * CAPITAL / price)
        if qty <= 0:
            continue
        prev = float(close.iloc[-2][code])
        flags = MARKET_FLAGS.get(code, {})
        orders.append(dict(
            symbol=code, name=str(names.get(code, "")), side="BUY",
            price=price, qty=qty, order_value=qty * price,
            prev_close=prev, last_price=price, tick_size=kr_tick(price),
            halted=flags.get("halted", False), vi_active=flags.get("vi", False),
            adv=krdata.adv(date, code),          # 최근 20일 평균거래량 → REG-010 시장충격 검사
        ))
    return {"orders": orders,
            "logs": [f"[dealer] {agent.name} 포트폴리오 → {len(orders)}개 매수 주문 생성"]}


def mo_review_node(state):
    """각 주문을 KRX 규정 M/O로 검사 (RAG 근거 + Evaluator)"""
    blotter, needs_human = [], False
    for o in state["orders"]:
        res = check_compliance(o)
        if not res.passed:
            verdict = "REJECTED"
        elif o["order_value"] >= HUMAN_VALUE or res.flags:
            verdict = "NEEDS_HUMAN"; needs_human = True
        else:
            verdict = "APPROVED"
        blotter.append({**o, "verdict": verdict,
                        "reasons": [v.reg_id for v in res.violations] or
                                   [f.reg_id for f in res.flags]})
    n_rej = sum(b["verdict"] == "REJECTED" for b in blotter)
    n_hum = sum(b["verdict"] == "NEEDS_HUMAN" for b in blotter)
    return {"blotter": blotter, "needs_human": needs_human,
            "logs": [f"[mo] 검사 완료: 반려 {n_rej} · 사람결재 {n_hum} · "
                     f"자동승인 {len(blotter)-n_rej-n_hum}"]}


def route_human(state):
    return "human" if state["needs_human"] else "execute"


def human_gate_node(state):
    """HITL — 고액 주문 일괄 사람 결재"""
    flagged = [b for b in state["blotter"] if b["verdict"] == "NEEDS_HUMAN"]
    verdict = interrupt({"ask": "고액 주문 사람 결재 필요", "count": len(flagged),
                         "orders": [(b["name"], b["order_value"]) for b in flagged]})
    new_blotter = []
    for b in state["blotter"]:
        if b["verdict"] == "NEEDS_HUMAN":
            b = {**b, "verdict": verdict}     # 결재 결과 반영
        new_blotter.append(b)
    return {"blotter": new_blotter, "decision": verdict,
            "logs": [f"[human] {len(flagged)}건 결재 결과: {verdict}"]}


def execute_node(state):
    filled = [b for b in state["blotter"] if b["verdict"] == "APPROVED"]
    notional = sum(b["order_value"] for b in filled)
    return {"logs": [f"[execute] 체결 {len(filled)}종목, 집행금액 {notional:,.0f}원"]}


def build_graph():
    g = StateGraph(OrchState)
    for n, fn in [("select", select_node), ("build_orders", build_orders_node),
                  ("mo_review", mo_review_node), ("human_gate", human_gate_node),
                  ("execute", execute_node)]:
        g.add_node(n, fn)
    g.add_edge(START, "select")
    g.add_edge("select", "build_orders")
    g.add_edge("build_orders", "mo_review")
    g.add_conditional_edges("mo_review", route_human,
                            {"human": "human_gate", "execute": "execute"})
    g.add_edge("human_gate", "execute")
    g.add_edge("execute", END)
    return g.compile(checkpointer=MemorySaver())


if __name__ == "__main__":
    close, meta = load_data()
    DATA.update(close=close, meta=meta,
                rets=daily_returns(close), mkt=market_return(daily_returns(close)))

    app = build_graph()
    cfg = {"configurable": {"thread_id": "kr-session-1"}}
    init = {"selected_agent": "", "selection_report": {}, "orders": [],
            "blotter": [], "needs_human": False, "decision": "", "logs": []}

    out = app.invoke(init, cfg)
    if "__interrupt__" in out:
        print(">> HITL: 고액 주문 사람 결재 대기 → 결재자 'APPROVED' 입력\n")
        out = app.invoke(Command(resume="APPROVED"), cfg)

    print("=" * 70)
    print(f"시니어 딜러 오케스트레이션 (실데이터) — 채택 룰: {out['selected_agent']}")
    print("=" * 70)
    print("\n[블로터] 종목 / 수량 / 단가 / 금액 / M/O 판정")
    for b in out["blotter"]:
        mark = {"APPROVED": "✓승인", "REJECTED": "✗반려", "NEEDS_HUMAN": "⧖결재"}.get(b["verdict"], b["verdict"])
        rs = f"  ({','.join(b['reasons'])})" if b["reasons"] else ""
        print(f"  {b['symbol']} {b['name']:<10} {b['qty']:>7,}주 @{b['price']:>8,} "
              f"= {b['order_value']:>14,}원  {mark}{rs}")

    filled = [b for b in out["blotter"] if b["verdict"] == "APPROVED"]
    print(f"\n체결: {len(filled)}종목 / 집행금액 {sum(b['order_value'] for b in filled):,.0f}원")
    print("\n[감사 로그]")
    for line in out["logs"]:
        print("  " + line)
