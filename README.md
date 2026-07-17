# AI 시니어 딜러 멀티에이전트 주식 트레이딩 시스템

시니어 딜러(Supervisor)가 여러 전략 agent를 경쟁시켜 최적 룰을 선정하고,
M/O가 KRX 규정을 검증한 뒤 사람 최종 승인을 거쳐 체결하는 자동 딜링 데스크.
코스피 실데이터(200종목, 2019~2026)로 동작한다.

동작 흐름:
  데이터 → 전략 4종 경쟁(종목선정+한도+매도규칙) → 선정 엔진(워크포워드+반감기 가중)
        → 시니어 딜러(주문 생성) → M/O 규정 검사(RAG) → 사람 최종 승인 → 체결

교재 매핑: 1주차(Tool 호출) · 2주차(LangGraph State/Node/Edge·Supervisor·HITL)
          · 3주차(RAG) · 4주차(MCP)

---

## 파일 구성

### 핵심 시스템
- `krdata.py`               데이터 로더·수익률·시장수익률·유니버스 스크리닝·베타
- `kr_agents.py`            전략 4종(모멘텀/평균회귀/돌파/시총상위10) + 3대 한도 + 매도규칙(exit_mask)
- `kr_backtest.py`          백테스트 + 워크포워드 선정 + 반감기 가중 + 변동성 유동손절
- `mo_engine.py`            M/O 규정 엔진 (RAG + KRX 규정 + LangGraph + HITL)
- `portfolio_orchestrator.py`  LangGraph 오케스트레이션 (선정→주문→M/O→결재→체결)
- `app.py`                  Flask 서버 (HTML 데모 + 실데이터 파이프라인 API)
- `trading_desk.html`       웹 딜링 데스크 UI (선정→주문→최종승인 팝업→체결→로그)

### 데이터 (실행에 필요)
- `kr_prices.csv`           코스피 200종목 일별 종가 (2019~2026)
- `kr_meta.csv`             종목 메타 (code, name, market, sector)
- `kr_shares.csv`           상장주식수 (시총 상위 바스켓용)

### 데이터 수집 스크립트 (본인 PC에서 실행)
- `fetch_kr_data.py`        시세·메타 수집 → kr_prices.csv, kr_meta.csv
- `fetch_shares.py`         상장주식수 수집 → kr_shares.csv
- `update_prices.py`        시세를 오늘까지 증분 갱신
- `opendart_fetch.py`       OpenDART 재무데이터 → kr_fundamental.csv (밸류 agent용, 선택)
- `append_year.py`          특정 연도 재무만 증분 추가
- `fetch_kr_fundamental.py` pykrx 펀더멘털 수집 (KRX 불안정, opendart 권장)

---

## 실행 방법

### 1) 설치
```
pip install pandas numpy flask langgraph scikit-learn matplotlib finance-datareader
```

### 2) 백테스트 (전략 선정 결과 + 차트)
```
python kr_backtest.py
```
→ 반감기 1·3·6개월 × 4전략 비교, 최고 조합 채택, kr_halflife_result.png 생성

### 3) 웹 딜링 데스크 데모
```
python sim_server.py            # http://localhost:5001
```
→ 브라우저에서 「딜링 실행」:
   선정 → 주문 → M/O 규정검사 → (근거+비교그래프) 최종 승인 팝업 → 체결 → 로그
   (시작 시 오늘자 시세 자동 갱신 시도. 끄려면 DESK_AUTOUPDATE=0)

### 4) LangGraph 오케스트레이션 (콘솔)
```
python portfolio_orchestrator.py
```

---

## 데이터 갱신

- 시세(매일 변함): `python update_prices.py` 또는 app.py 자동 갱신
- 재무(분기·연 단위): 새 사업보고서 나올 때 `python append_year.py`
- 상장주식수: 거의 안 변함, 필요 시 `python fetch_shares.py` 1회

## 주의사항

- 회사/기관 네트워크에서 SSL 오류 시: 수집 스크립트는 기본으로 검증을 우회한다
  (공개 시세만 받음). 검증을 유지하려면 `SECURE_SSL=1`로 실행.
- 이 데이터 구간은 역대급 상승장이라, 위험조정성과(샤프)·낙폭 기준으로 해석할 것.
- 백테스트는 look-ahead 차단·거래비용·워크포워드 검증을 반영한다.

## 주요 파라미터 (kr_backtest.py 상단)

- `STOP_LOSS=0.20`, `TRAIL_STOP=0.30`   고정 손절/트레일링 (완화 세팅)
- `run_portfolio_bt(..., dynamic_stop=True, k_stop=5, k_trail=8)`  변동성 유동손절
- `HALFLIVES={"1개월":21,"3개월":63,"6개월":126}`  반감기 후보
- `max_drawdown_limit=0.30` (kr_agents.py)  트레일링 고점 대비 낙폭 정지
