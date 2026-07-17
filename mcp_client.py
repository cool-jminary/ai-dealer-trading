"""
M/O 규정 MCP 클라이언트 + 도구선택 에이전트 (4주차)

교재의 MCP 클라이언트 '네 박자'를 구현한다:
  ① 연결(connect)   : mcp_server.py 를 STDIO 자식 프로세스로 띄워 통신줄 열기
  ② 초기화(initialize): 핸드셰이크(첫 인사)
  ③ 목록(tools/list) : 서버에 등록된 도구 목록 받기
  ④ 호출(tools/call) : 고른 도구를 인자와 함께 실행

에이전트의 '도구 선택':
  LLM(ChatGPT 4o)에게 tools/list 로 받은 도구들의 이름·설명(docstring)을 보여주고,
  주어진 주문 상황에 어떤 도구를 어떤 인자로 부를지 스스로 고르게 한다(진짜 tool calling).
  OPENAI_API_KEY 가 없으면 규칙 기반으로 도구를 선택(폴백)한다.

사용:
  import asyncio, mcp_client
  result = asyncio.run(mcp_client.review_order({...주문...}))
"""
import os
import sys
import json
import asyncio

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

import llm

_HERE = os.path.dirname(os.path.abspath(__file__))
_SERVER = StdioServerParameters(command=sys.executable, args=[os.path.join(_HERE, "mcp_server.py")])


class MCPToolClient:
    """재사용 클라이언트 — 연결·초기화를 한 번만 하고 여러 번 호출한다(교재 NB03)."""
    def __init__(self):
        self._cm = None
        self.session: ClientSession | None = None
        self.tools = []                 # tools/list 결과 (이름·설명·입력스키마)

    async def connect(self):
        self._cm = stdio_client(_SERVER)                 # ① 연결(통신줄 열기)
        self._read, self._write = await self._cm.__aenter__()
        self._sess_cm = ClientSession(self._read, self._write)
        self.session = await self._sess_cm.__aenter__()
        await self.session.initialize()                  # ② 초기화(첫 인사)
        resp = await self.session.list_tools()           # ③ 목록(tools/list)
        self.tools = [{"name": t.name, "description": t.description or "",
                       "input_schema": t.inputSchema} for t in resp.tools]
        return self.tools

    async def call(self, name, arguments):               # ④ 호출(tools/call)
        res = await self.session.call_tool(name, arguments)
        out = []
        for c in res.content:
            txt = getattr(c, "text", None)
            if txt is None:
                continue
            try:
                out.append(json.loads(txt))
            except Exception:
                out.append(txt)
        return out[0] if len(out) == 1 else out

    async def disconnect(self):
        try:
            await self._sess_cm.__aexit__(None, None, None)
            await self._cm.__aexit__(None, None, None)
        except Exception:
            pass


# ----------------------------------------------------------------------
# 에이전트: 도구 목록을 보고 '어떤 도구를 쓸지' 선택
# ----------------------------------------------------------------------
def _select_tools_llm(order, tools):
    """LLM이 도구 설명(docstring)을 읽고, 이 주문에 쓸 도구와 인자를 고른다.
    반환: [{"tool": name, "arguments": {...}}]  (JSON)"""
    tool_desc = "\n".join(f"- {t['name']}: {t['description'].strip().splitlines()[0]}"
                          for t in tools)
    system = ("너는 국민은행 주식 딜링의 M/O(미들·백오피스) 준법 에이전트다. 아래 MCP 도구 목록을 보고, "
              "주어진 주문을 심사하기 위해 호출할 도구와 인자를 고른다. "
              "규정 위반 가능성 확인이 필요하면 search_regulations를, 대량주문 시장충격 확인이 "
              "필요하면 check_market_impact를 사용한다. 필요하면 둘 다 쓸 수 있다. "
              "반드시 아래 JSON 스키마로만 답한다: "
              '{"calls":[{"tool":"도구명","arguments":{...}}]}')
    user = (f"[사용 가능 도구]\n{tool_desc}\n\n"
            f"[주문]\n종목 {order.get('name','')}({order.get('symbol','')}), "
            f"{order.get('side','BUY')}, 가격 {order.get('price')}, 수량 {order.get('qty')}주, "
            f"주문금액 {order.get('order_value','?')}, "
            f"상한가 여부/거래정지/VI 등 상태: halted={order.get('halted')}, vi={order.get('vi_active')}, "
            f"ADV(20일평균거래량)={order.get('adv')}\n\n"
            "이 주문 심사에 필요한 도구 호출을 JSON으로.")
    data = llm.chat_json(system, user)
    if not data or "calls" not in data:
        return None
    return data["calls"]


def _select_tools_rule(order, tools):
    """키 없을 때: 규칙으로 도구 선택. 규정검색은 항상, 대량이면 시장충격도."""
    calls = [{"tool": "search_regulations",
              "arguments": {"query": _order_to_query(order), "k": 3}}]
    adv = order.get("adv")
    qty = order.get("qty", 0)
    if adv and adv > 0:                       # ADV 있으면 시장충격도 확인
        calls.append({"tool": "check_market_impact",
                      "arguments": {"order_qty": int(qty), "adv": float(adv)}})
    return calls


def _order_to_query(order):
    bits = []
    if order.get("halted"):    bits.append("거래정지 정리매매 종목")
    if order.get("vi_active"): bits.append("변동성완화장치 VI")
    bits.append(f"{order.get('side','BUY')} 주문 상한가 가격제한폭")
    return " ".join(bits)


async def review_order(order):
    """주문 하나를 MCP 도구로 심사한다.
    반환: {tool_selection_by, calls:[{tool,arguments,result}], tools_available:[...]}"""
    client = MCPToolClient()
    tools = await client.connect()                       # ①②③ 연결·초기화·목록
    try:
        if llm.available():
            calls = _select_tools_llm(order, tools) or _select_tools_rule(order, tools)
            by = "LLM(ChatGPT 4o)" if llm.available() else "규칙"
        else:
            calls = _select_tools_rule(order, tools)
            by = "규칙"
        results = []
        for c in calls:                                  # ④ 선택한 도구들 실행
            name = c.get("tool"); args = c.get("arguments", {})
            if name not in {t["name"] for t in tools}:   # 서버에 없는 도구는 스킵
                continue
            res = await client.call(name, args)
            results.append({"tool": name, "arguments": args, "result": res})
        return {"tool_selection_by": by,
                "tools_available": [t["name"] for t in tools],
                "calls": results}
    finally:
        await client.disconnect()


def review_order_sync(order):
    """동기 래퍼 — Flask 등에서 그냥 호출."""
    return asyncio.run(review_order(order))


if __name__ == "__main__":
    # 서브프로세스 모드: --order '<json>' → 심사 결과를 JSON 한 줄로 출력(Flask가 호출)
    if len(sys.argv) >= 3 and sys.argv[1] == "--order":
        order = json.loads(sys.argv[2])
        print(json.dumps(review_order_sync(order), ensure_ascii=False))
        sys.exit(0)

    # 데모: 상한가 초과 + 대량 주문
    demo = {"symbol": "005930", "name": "삼성전자", "side": "BUY",
            "price": 100000, "qty": 500000, "order_value": 5_000_000_000,
            "halted": False, "vi_active": True, "adv": 1_000_000}
    out = review_order_sync(demo)
    print(f"도구 선택 주체: {out['tool_selection_by']}")
    print(f"사용 가능 도구(tools/list): {out['tools_available']}\n")
    for c in out["calls"]:
        print(f"▶ tools/call: {c['tool']}({c['arguments']})")
        print(f"   → {json.dumps(c['result'], ensure_ascii=False)[:200]}\n")
