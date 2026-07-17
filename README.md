# AI 시니어 딜러 멀티에이전트 주식 트레이딩 시스템

시니어 딜러(Supervisor)가 9개 전략 봇을 경쟁시켜 최적 룰을 선정하고,
M/O가 KRX 규정(+손실한도)을 검증한 뒤 사람 승인을 거쳐 체결하는 자동 딜링 데스크.
코스피 실데이터(200종목, 2019~2026)로 동작하며, LLM(ChatGPT 4o)이 규정 판단·근거 설명·매매 소명을,
MCP가 규정·시장충격 도구 심사를 담당한다.

동작 흐름:
  데이터 → 9개 전략 봇 경쟁 → 선정 엔진(워크포워드 + 반감기 가중) → 시니어 딜러(주문)
        → M/O 규정 검사(RAG + LLM + MCP) → 사람 최종 승인(HITL) → 체결
        ※ M/O 반려 또는 사람 거절 시 → 다시 딜러로 돌아가 '다음 순위 종목'으로 재심사(루프)

교재 매핑: 1주차(Tool 호출) · 2주차(LangGraph/Supervisor/HITL) · 3주차(RAG) · 4주차(MCP) · LLM 실사용

---

## 파일 구성

### 핵심 시스템
- krdata.py                 데이터 로더 · OHLC 패널 · 수익률 · 유니버스 · 베타 · ADV · 하락장 방어(현금화)
- kr_agents.py              전략 9종 + 3대 한도 + 매도규칙(exit_mask)
- kr_backtest.py            백테스트 + 워크포워드 선정 + 반감기 + ATR 손절 + Buy&Hold 엔진 + 방어(defend)
- kr_buyhold.py             Buy&Hold 2종(순수/피라미딩) 단독 비교 스크립트
- mo_engine.py              M/O 규정 엔진 (RAG + KRX 규정 + 손실한도 + LLM 판단 + MCP 경유 + LangGraph/HITL)
- mcp_server.py             [4주차 MCP] FastMCP 도구 서버 (STDIO) — 규정검색·시장충격 도구 2종
- mcp_client.py             [4주차 MCP] MCP 클라이언트 + 도구선택 에이전트 (tools/list → 선택 → tools/call)
- llm.py                    OpenAI(ChatGPT 4o) 클라이언트 (키 없으면 규칙 폴백)
- portfolio_orchestrator.py LangGraph 오케스트레이션
- sim_server.py             ★일자 진행 시뮬레이터 서버 (포트 5001) — 메인 데모
- sim_desk.html             ★시뮬레이터 UI (9봇 시각화·손실한도·거래 소명·MCP 규정 심사)
- app.py / trading_desk.html  (참고) 파이프라인 데스크 초기 버전 — 발표엔 미사용, 실행은 가능

### 전략 9종 (make_agents)
  1 모멘텀 · 2 평균회귀 · 3 돌파 · 4 시총상위10 · 5 저변동성
  6 52주신고가 · 7 거래량 · 8 BuyHold순수 · 9 BuyHold피라미딩

### 데이터 (실행에 필요)
- kr_prices.csv 종가 | kr_meta.csv 메타 | kr_shares.csv 상장주식수
- (선택) kr_open/high/low.csv OHLC · kr_volume.csv 거래량 → 있으면 돌파·ATR·거래량봇·ADV 자동 활성화

### 데이터 수집 (본인 PC에서 실행)
- fetch_kr_data.py 시세(OHLC)·메타 | fetch_shares.py 상장주식수 | update_prices.py 증분 갱신
- opendart_fetch.py / append_year.py / fetch_kr_fundamental.py  재무(밸류 확장용)

---

## 실행 방법

설치
    pip install -r requirements.txt

백테스트
    python kr_backtest.py       # 9종 선정 백테스트
    python kr_buyhold.py        # Buy&Hold 2종 vs 시장 비교

MCP 도구 서버·클라이언트 (4주차)
    python mcp_client.py        # tools/list → 도구 선택 → tools/call 전 과정 시연
                                # (내부에서 mcp_server.py 를 STDIO 자식 프로세스로 자동 기동)

■ 메인 데모 — 일자 진행 시뮬레이터 (포트 5001)
    python sim_server.py        # 첫 기동 시 최신 시세 증분 갱신 + 2023~2026.5 자동 사전운용(~1분)
    - 왼쪽: 9개 전략 봇 (선정된 봇 활성화 + 말풍선으로 매매)
    - 6월부터 '다음 영업일'로 진행 · 매수/매도 확인(총액)·딜러 거절→다음 순위 대체
    - 운용한도 300억 · 연 정산 · 손실한도(일 평가액의 10% / 월 월초의 30%, 축소 시 평가액의 1/3)
    - 변동성 적응형 하락장 방어(현금화) · 거래번호(T-YYYYMMDD-NNNN) 감사
    - 'MCP 규정 심사' 패널: 거래번호 입력 → 에이전트가 MCP 도구를 골라 심사(tools/list→call)
    ※ 발표·시연은 이 시뮬레이터 하나로 진행한다.

(참고) 파이프라인 데스크 — app.py / trading_desk.html (포트 5000)
    선정 → 주문 → M/O(RAG+LLM) → 근거+비교그래프 최종 승인 → 체결을 '한 번에' 보여주는 초기 버전.
    시뮬레이터에 기능이 모두 포함돼 있어 발표에는 쓰지 않는다. 필요 시 python app.py 로 실행 가능.

LLM(ChatGPT 4o) 활성화 — 실사용 4곳 + MCP 도구선택
    export OPENAI_API_KEY=sk-...          # 실제 키
    # 회사망 SSL 이슈:  export OPENAI_INSECURE_SSL=1
  (1) M/O 규정 판단  (2) 선정 근거 설명  (3) 매매 근거 설명  (4) 거래 소명 질의  (+ MCP 도구 선택)
  → 키 없으면 전부 규칙 기반으로 자동 폴백(데모는 그대로 동작).

---

## MCP (4주차) — 무엇을 어떻게 구현했나

MCP(Model Context Protocol)는 'LLM과 도구의 연결 방식을 통일한 표준 약속'이다.
공개 서버로 해결되는 것(환율·주가)과 달리, '사내 규정 검색'은 우리만 가진 내부 자료라 직접 서버로 만든다.

- mcp_server.py : FastMCP 서버. 함수 위에 @mcp.tool() 표식을 붙여 도구로 공개.
  docstring이 곧 LLM이 읽는 사용설명서. STDIO(표준입출력)로 통신.
    · search_regulations(query, k)      — KRX 규정을 키워드로 검색(RAG) → 관련 조항 반환
    · check_market_impact(order_qty, adv) — 주문수량÷ADV로 시장충격 판정(정상/경고/반려)
- mcp_client.py : MCP 클라이언트의 네 박자(연결 → 초기화 → 목록 → 호출)를 구현.
    · tools/list 로 서버의 도구 목록을 받고
    · 에이전트가 '어떤 도구를 쓸지' 선택 — LLM이 docstring을 읽고 판단(진짜 tool calling),
      키 없으면 규칙으로 선택(폴백)
    · tools/call 로 고른 도구를 실행하고 결과를 모음
- 연결: mo_engine.check_compliance(order, use_mcp=True) 로 M/O 심사에서 MCP 경유 가능.
  시뮬레이터는 /api/sim/mcp_review (거래번호 입력)로 특정 주문을 MCP 도구로 심사해 보여준다.

핵심: 도구를 표준 인터페이스(MCP)로 공개 → 어느 에이전트/프레임워크든 tools/list·tools/call
      같은 방식으로 재사용. 연결이 N×M에서 N+M으로 줄어든다.

---

## 규정 용어 — ADV와 VI

- ADV (Average Daily Volume, 평균거래량)
    최근 20영업일 평균 거래량(주). '이 종목이 하루에 보통 몇 주 거래되는가'의 척도.
    주문수량이 ADV에 비해 크면 시장충격(내가 사면서 값을 밀어올림)이 커진다.
    규정 REG-010: 주문수량이 ADV의 10% 초과 → 경고, 25% 초과 → 반려(분할·축소 요구).
    MCP 도구 check_market_impact 가 order_qty÷adv 로 이 비율을 계산해 판정한다.

- VI (Volatility Interruption, 변동성완화장치)
    개별 종목의 가격이 순간적으로 급변하면 2분간 단일가매매로 전환해 과열을 식히는 장치.
    VI가 발동된 동안에는 '즉시 체결'을 요구하는 주문은 접수되지 않는다.
    규정 REG-005. 주문 심사에서 VI 발동 종목의 즉시체결 주문은 반려 대상이 된다.

기타 주요 규정: REG-001 가격제한폭(전일 종가 ±30%) · REG-002 거래정지/정리매매 종목 ·
  REG-003 공매도(무차입 금지·업틱룰) · REG-004 자전거래 금지 · REG-007 호가가격단위(틱).

---

## 핵심 실험 결론 (발표 알맹이)
- 국면에 따라 선정 전략이 바뀐다 (넓은 구간→평균회귀, 급등장→모멘텀/돌파).
- "똑똑하게 갈아타기"는 대부분 실패: 몰아주기·재선정·유사국면검색·피라미딩 모두
  단일 전략/Buy&Hold를 못 이김 → 잦은 개입은 뒷북(whipsaw)·비용을 낳음.
- 워크포워드의 정직한 한계: 학습이 등락장인데 검증이 급등장이면 선정이 빗나갈 수 있음.
- 매도 규칙 다이얼: 타이트(방어) ↔ 완화(공격), ATR 유동 손절이 모멘텀에 특히 효과.

## 주의
- 데이터 구간이 역대급 상승장 → 위험조정성과(샤프)·낙폭으로 해석할 것.
- OHLC/거래량 CSV가 없으면 돌파는 종가, 거래량봇은 빈 선택으로 폴백(코드 수정 불필요).
- 시뮬레이터의 손익 계산은 전일 종가 기준이라 하루치 갭만큼 낙관적(시가 체결로 정밀화 가능).