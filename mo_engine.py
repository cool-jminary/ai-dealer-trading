"""
M/O(미들오피스) 규정 엔진 — 시니어 딜러 주문의 컴플라이언스 관문

교재 적용:
  - 3주차 RAG      : 사내 규정 조항을 임베딩·검색해 '출처를 제시'하며 판단
  - 4주차 MCP      : 규정 검색기를 search_documents 도구로 감싸 표준화·재사용
  - 2주차 LangGraph: 메이커(시니어 딜러)–체커(M/O) 분리 + Evaluator + 조건부 엣지
                     고액 주문은 interrupt()로 사람 결재(HITL)

그래프 흐름:
  START → retrieve(규정검색) → compliance(Evaluator) → [조건부 엣지]
        → approve(승인) / reject(반려) / human(사람 결재) → END
"""
import operator
from dataclasses import dataclass, field, asdict
from typing import TypedDict, Annotated

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command

import llm   # OpenAI(ChatGPT 4o) 클라이언트. 키 없으면 available()=False → 규칙 폴백

HUMAN_VALUE = 3_000_000_000   # 30억 이상 주문은 딜러(사람) 결재

# 시장충격·유동성(ADV 대비 주문규모) 임계값 — REG-010
ADV_FLAG_RATIO = 0.10         # 주문수량이 ADV의 10% 초과 → 시장충격 경고(플래그)
ADV_BLOCK_RATIO = 0.25        # 25% 초과 → 반려(분할·축소 요구)


# ======================================================================
# 사내 규정 문서 (RAG 지식베이스) — 각 조항 = 청크 1개
# ======================================================================
REGULATIONS = [
    {"id": "REG-001", "title": "가격제한폭(상하한가)",
     "text": "유가증권·코스닥 상장주식의 주문가격은 전일 종가 대비 상하 30% 이내여야 하며, "
             "이를 벗어난 주문은 거래소가 접수하지 않는다."},
    {"id": "REG-002", "title": "거래정지·정리매매 종목",
     "text": "거래정지, 관리종목 정리매매, 상장폐지 절차 진행 종목에 대한 신규 주문은 접수할 수 없다."},
    {"id": "REG-003", "title": "공매도 규정",
     "text": "차입하지 않은 무차입 공매도는 금지된다. 공매도 호가는 직전 체결가 이하로 제출할 수 없다(업틱룰). "
             "공매도 과열종목으로 지정된 경우 익영업일 공매도가 제한된다."},
    {"id": "REG-004", "title": "자전거래 금지",
     "text": "동일인 또는 동일 계좌군이 매수와 매도를 동시에 제출하여 시세나 거래량을 오인하게 하는 "
             "자전성 거래는 시세조종 우려로 금지된다."},
    {"id": "REG-005", "title": "변동성완화장치(VI)",
     "text": "개별종목 VI가 발동된 동안에는 2분간 단일가매매로 전환되며, 즉시 체결을 요구하는 주문은 접수되지 않는다."},
    {"id": "REG-006", "title": "대량보유보고(5%룰)",
     "text": "본인과 특별관계자의 보유 지분이 발행주식총수의 5% 이상이 되면 5영업일 이내에 "
             "대량보유 상황을 금융위·거래소에 보고해야 한다."},
    {"id": "REG-007", "title": "호가가격단위(틱)",
     "text": "주문가격은 가격대별로 정해진 호가가격단위의 정수배여야 한다."},
    {"id": "REG-008", "title": "일일 손실한도(-30억)",
     "text": "데스크의 당일 실현·평가 손익이 일일 손실한도(-30억원)에 도달하면 리스크 관리 규정에 따라 "
             "익영업일부터 1주간(5영업일) 신규 매매를 정지한다. 보유 포지션은 유지한다."},
    {"id": "REG-009", "title": "월간 손실한도(-100억)·운용한도 축소",
     "text": "당월 초 대비 손익이 월간 손실한도(-100억원)에 도달하면 운용한도를 300억에서 100억으로 축소하고, "
             "초과 포지션(손실 큰 종목 우선)을 매도하며, 1개월간(21영업일) 신규 매매를 정지한다. "
             "정지 해제 시 운용한도는 300억으로 복구된다."},
    {"id": "REG-010", "title": "시장충격·유동성(ADV 대비 주문규모)",
     "text": "주문수량이 최근 20영업일 평균거래량(ADV)의 10%를 초과하면 시장충격 우려로 경고하고, "
             "25%를 초과하면 주문을 반려하여 분할·축소를 요구한다. 대량 주문은 체결 시 가격을 크게 "
             "밀어올려(내려) 불리한 체결과 시세 왜곡을 유발할 수 있다."},
]
REG_BY_ID = {d["id"]: d for d in REGULATIONS}


# ======================================================================
# RAG 검색기 (파싱=텍스트 / 청킹=조항단위 / 임베딩=TF-IDF / 리트리버=코사인)
#   * 실무에서는 3~4주차대로 ChromaDB + 임베딩 모델을 쓴다. 여기선 오프라인 데모용.
# ======================================================================
class RegulationRetriever:
    def __init__(self, docs):
        self.docs = docs
        self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
        self.mat = self.vec.fit_transform([d["title"] + " " + d["text"] for d in docs])

    def search(self, query, k=3):
        qv = self.vec.transform([query])
        sims = cosine_similarity(qv, self.mat)[0]
        idx = sims.argsort()[::-1][:k]
        return [{**self.docs[i], "score": round(float(sims[i]), 3)} for i in idx]


_retriever = RegulationRetriever(REGULATIONS)


def search_documents(query: str, k: int = 3):
    """[MCP 도구 · 4주차] 규정 검색기를 표준 도구로 감싼 것.
    어느 에이전트든 같은 방식으로 규정을 조회한다(재사용·표준화·역할분리)."""
    return _retriever.search(query, k)


# ======================================================================
# 구조화 결과 (Evaluator 출력) — 통과여부·위반·플래그·피드백
# ======================================================================
@dataclass
class Violation:
    reg_id: str
    title: str
    message: str
    clause: str          # 출처: 규정 조항 원문


@dataclass
class MOResult:
    passed: bool
    violations: list = field(default_factory=list)
    flags: list = field(default_factory=list)     # 차단 아님(보고 대상 등)
    feedback: str = ""
    llm: dict = None                               # LLM(ChatGPT 4o) 규정 판단 {verdict,reason,regs}
    mcp: dict = None                               # MCP 도구선택·호출 결과 {tool_selection_by,calls}

    def as_dict(self):
        return {"passed": self.passed,
                "violations": [asdict(v) for v in self.violations],
                "flags": [asdict(v) for v in self.flags],
                "feedback": self.feedback,
                "llm": self.llm}


def _clause(rid):
    d = REG_BY_ID[rid]
    return d["title"], d["text"]


# ======================================================================
# 컴플라이언스 검사 (실제 규정 룰) — 각 위반은 조항을 출처로 인용
# ======================================================================
def check_compliance(o: dict, use_llm: bool = False, use_mcp: bool = False) -> MOResult:
    v, flags = [], []

    if o.get("halted"):                                              # 거래정지
        t, c = _clause("REG-002"); v.append(Violation("REG-002", t, "거래정지 종목 신규 주문 불가", c))

    pc = o["prev_close"]                                             # 가격제한폭
    upper, lower = pc * 1.30, pc * 0.70
    if o["price"] > upper or o["price"] < lower:
        t, c = _clause("REG-001")
        v.append(Violation("REG-001", t,
                 f"주문가 {o['price']:,.0f}원이 제한폭({lower:,.0f}~{upper:,.0f}원) 이탈", c))

    if o.get("vi_active"):                                           # VI 발동
        t, c = _clause("REG-005"); v.append(Violation("REG-005", t, "VI 발동 중 즉시체결 주문 불가", c))

    if o["side"] == "SELL" and o.get("is_short"):                    # 공매도
        if not o.get("borrowed"):
            t, c = _clause("REG-003"); v.append(Violation("REG-003", t, "무차입 공매도 금지", c))
        if o["price"] < o.get("last_price", o["price"]):
            t, c = _clause("REG-003"); v.append(Violation("REG-003", t, "업틱룰 위반(직전가 미만 공매도호가)", c))
        if o.get("overheated_short"):
            t, c = _clause("REG-003"); v.append(Violation("REG-003", t, "공매도 과열종목 제한", c))

    if o.get("self_cross"):                                          # 자전거래
        t, c = _clause("REG-004"); v.append(Violation("REG-004", t, "동일 계좌군 자전성 거래 금지", c))

    tick = o.get("tick_size", 0)                                     # 호가단위
    if tick and abs(o["price"] / tick - round(o["price"] / tick)) > 1e-9:
        t, c = _clause("REG-007"); v.append(Violation("REG-007", t, f"호가단위({tick:,}원) 미준수", c))

    own = o.get("resulting_ownership", 0.0)                          # 대량보유(플래그)
    if own >= 0.05:
        t, c = _clause("REG-006")
        flags.append(Violation("REG-006", t, f"주문 후 보유 {own*100:.1f}% → 대량보유보고 대상", c))

    adv = o.get("adv")                                               # 최근 20일 평균거래량(주)
    qty = o.get("qty", 0)                                            # 주문수량
    if adv and adv > 0 and qty:                                      # 시장충격·유동성(REG-010)
        ratio = qty / adv
        t, c = _clause("REG-010")
        if ratio > ADV_BLOCK_RATIO:                                  # 과대 주문 → 반려
            v.append(Violation("REG-010", t,
                     f"주문수량이 ADV의 {ratio*100:.0f}%(> {ADV_BLOCK_RATIO*100:.0f}%) → 시장충격 과다, 분할·축소 필요", c))
        elif ratio > ADV_FLAG_RATIO:                                 # 경고(플래그)
            flags.append(Violation("REG-010", t,
                     f"주문수량이 ADV의 {ratio*100:.0f}% → 시장충격 경고", c))

    passed = len(v) == 0
    if passed:
        fb = "규정 위반 없음. 체결 진행 가능" + (" (대량보유보고 필요)" if flags else "")
    else:
        fb = "위반 " + ", ".join(x.reg_id for x in v) + " → 주문 반려, 시니어 딜러에 사유 반환"
    result = MOResult(passed, v, flags, fb)

    # ── MCP(4주차): 에이전트가 MCP 도구를 골라 규정검색·시장충격 조회 ──
    if use_mcp:
        try:
            import mcp_client
            result.mcp = mcp_client.review_order_sync(o)
        except Exception as e:
            result.mcp = {"error": str(e)}

    # ── LLM(ChatGPT 4o) 규정 판단 부착 (하드룰은 위에서 이미 안전하게 확정) ──
    if use_llm:
        result.llm = judge_order_llm(o, result)
    return result


def judge_order_llm(o: dict, base: MOResult):
    """LLM이 관련 규정(RAG 검색)을 근거로 주문 위반 여부를 판정·설명.
    하드룰 결과(base)는 이미 안전하게 확정돼 있고, LLM은 근거 설명과 추가 정황을 담당한다.
    키 없거나 실패하면 None(호출부는 규칙 결과만 사용)."""
    if not llm.available():
        return None
    regs = search_documents(f"{o.get('name','')} {o.get('side','')} 주문 가격 {o.get('price')}", k=3)
    reg_text = "\n".join(f"[{r['id']}] {r['title']}: {r['text']}" for r in regs)
    rule_find = "; ".join(f"{x.reg_id} {x.message}" for x in base.violations) or "규칙 검사상 하드룰 위반 없음"
    system = ("너는 국민은행 주식 딜링데스크의 미들·백오피스(M/O) 규정 심사역이다. "
              "주어진 주문과 KRX 규정 조항을 근거로 위반 여부를 판정하고 근거를 한국어 2문장 이내로 설명한다. "
              "거래정지·가격제한폭·VI 등 명백한 하드룰 위반은 그대로 반영하고, "
              "자전거래·시세조종 정황 등 판단이 필요한 부분도 지적한다. "
              '반드시 JSON만 출력한다: {"verdict":"PASS"|"REJECT","reason":"...","regs":["REG-00X"]}')
    user = (f"[주문]\n종목 {o.get('name')}({o.get('symbol')}) · {o.get('side')} · "
            f"주문가 {o.get('price'):,}원 · 전일종가 {o.get('prev_close')}\n"
            f"거래정지={o.get('halted')} · VI발동={o.get('vi_active')} · 호가단위={o.get('tick_size')}\n\n"
            f"[관련 규정(RAG 검색)]\n{reg_text}\n\n[규칙엔진 1차 결과]\n{rule_find}")
    return llm.chat_json(system, user, max_tokens=220)


# ======================================================================
# LangGraph : State / Node / Edge
# ======================================================================
class MOState(TypedDict):
    order: dict
    retrieved: list
    result: dict
    decision: str
    logs: Annotated[list, operator.add]     # 누적 Reducer (감사 로그)


def retrieve_node(state):                    # RAG 규정 검색
    o = state["order"]
    query = (f"{o['side']} {'공매도' if o.get('is_short') else ''} 가격제한 "
             f"거래정지 VI 자전거래 대량보유 {o['symbol']}")
    hits = search_documents(query, k=3)
    return {"retrieved": hits,
            "logs": [f"[retrieve] RAG 규정 {len(hits)}건 검색: "
                     + ", ".join(h["id"] for h in hits)]}


def compliance_node(state):                  # Evaluator(체커)
    res = check_compliance(state["order"])
    return {"result": res.as_dict(),
            "logs": [f"[compliance] passed={res.passed} 위반={len(res.violations)}건"]}


def route(state):                            # 조건부 엣지
    res = state["result"]
    if not res["passed"]:
        return "reject"
    if state["order"].get("order_value", 0) >= HUMAN_VALUE or res["flags"]:
        return "human"
    return "approve"


def approve_node(state):
    return {"decision": "APPROVED", "logs": ["[decision] 승인 → 체결 진행"]}


def reject_node(state):
    return {"decision": "REJECTED", "logs": ["[decision] 반려 → 시니어 딜러에 피드백 반환"]}


def human_node(state):                       # HITL — 고액/보고대상 주문
    verdict = interrupt({"ask": "M/O 사람 결재 필요",
                         "order": state["order"], "result": state["result"]})
    return {"decision": verdict, "logs": [f"[decision] 사람 결재 결과: {verdict}"]}


def build_mo_graph():
    g = StateGraph(MOState)
    g.add_node("retrieve", retrieve_node)
    g.add_node("compliance", compliance_node)
    g.add_node("approve", approve_node)
    g.add_node("reject", reject_node)
    g.add_node("human", human_node)
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "compliance")
    g.add_conditional_edges("compliance", route,
                            {"approve": "approve", "reject": "reject", "human": "human"})
    for n in ("approve", "reject", "human"):
        g.add_edge(n, END)
    return g.compile(checkpointer=MemorySaver())


# ======================================================================
# 데모
# ======================================================================
def run_order(app, order, thread_id, approver="APPROVED"):
    cfg = {"configurable": {"thread_id": thread_id}}
    init = {"order": order, "retrieved": [], "result": {}, "decision": "", "logs": []}
    out = app.invoke(init, cfg)
    if "__interrupt__" in out:                       # 사람 결재 대기 → 결재자 입력 후 재개
        print(f"    · HITL 정지: 사람 결재 대기 → 결재자 '{approver}' 입력")
        out = app.invoke(Command(resume=approver), cfg)
    return out


def show(order, out):
    r = out["result"]
    print(f"\n[{order['symbol']}] {order['side']}  {order['price']:,.0f}원 x {order['qty']:,}주"
          f"  (주문금액 {order.get('order_value',0):,.0f})")
    print(f"    RAG 검색: {', '.join(h['id'] for h in out['retrieved'])}")
    print(f"    판정: {out['decision']}   | {r['feedback']}")
    for viol in r["violations"]:
        print(f"      ✗ {viol['reg_id']} {viol['title']}: {viol['message']}")
        print(f"          근거: {viol['clause'][:45]}…")
    for fl in r["flags"]:
        print(f"      ⚑ {fl['reg_id']} {fl['title']}: {fl['message']}")


if __name__ == "__main__":
    app = build_mo_graph()

    orders = [
        # 1) 정상 매수 → 승인
        dict(symbol="005930", side="BUY", price=71_000, qty=100, prev_close=70_000,
             last_price=71_000, tick_size=100, order_value=7_100_000),
        # 2) 상한가 초과 매수 → 반려(REG-001)
        dict(symbol="000660", side="BUY", price=150_000, qty=50, prev_close=100_000,
             last_price=110_000, tick_size=1000, order_value=7_500_000),
        # 3) 무차입 공매도 → 반려(REG-003)
        dict(symbol="035420", side="SELL", price=180_000, qty=30, prev_close=185_000,
             last_price=182_000, is_short=True, borrowed=False, tick_size=500,
             order_value=5_400_000),
        # 4) 거래정지 종목 → 반려(REG-002)
        dict(symbol="900110", side="BUY", price=3_200, qty=1000, prev_close=3_200,
             last_price=3_200, halted=True, tick_size=1, order_value=3_200_000),
        # 5) 고액 정상 주문 → 사람 결재(HITL) 후 승인
        dict(symbol="005930", side="BUY", price=71_000, qty=20_000, prev_close=70_000,
             last_price=71_000, tick_size=100, order_value=1_420_000_000),
        # 6) ADV 대비 과대 주문 → 반려(REG-010 시장충격, qty/ADV=50%)
        dict(symbol="000660", side="BUY", price=110_000, qty=500_000, prev_close=110_000,
             last_price=110_000, tick_size=1000, order_value=55_000_000_000, adv=1_000_000),
    ]

    print("=" * 64)
    print("M/O 규정 엔진 — 시니어 딜러 주문 컴플라이언스 검사")
    print("=" * 64)
    for i, o in enumerate(orders, 1):
        out = run_order(app, o, thread_id=f"order-{i}",
                        approver="APPROVED" if i == 5 else "APPROVED")
        show(o, out)