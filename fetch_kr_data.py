"""
한국 주식 실데이터 수집 스크립트  (본인 PC에서 실행)

이 환경(Claude 샌드박스)은 KRX/네이버/야후에 접근이 막혀 있어 한국 실데이터를
직접 받지 못합니다. 대신 이 스크립트를 '본인 PC'에서 한 번 돌리면
아래 두 CSV가 만들어지고, 그걸 업로드하면 본격 버전 백테스트가 그대로 돕니다.

설치:
    pip install finance-datareader pandas tqdm certifi

산출물:
    kr_prices.csv   — 날짜 x 종목코드 (종가 close 패널) ※ 기존 호환
    kr_open.csv     — 시가(Open) 패널
    kr_high.csv     — 고가(High) 패널
    kr_low.csv      — 저가(Low) 패널
    kr_volume.csv   — 거래량(Volume) 패널
    kr_meta.csv     — 종목코드, 종목명, 시장(KOSPI/KOSDAQ), 섹터(업종)

OHLC를 넣으면 돌파(장중 고가 돌파)·ATR 손절·시가 체결이 정교해지고,
거래량을 넣으면 '거래량 실린 돌파'만 인정하는 확인 필터와 ADV(평균거래량) 유동성 규정에 쓰인다.
기존 코드는 kr_prices.csv(종가)만으로도 그대로 동작하고, OHLC·거래량 파일이 있으면 자동 활용된다.

※ macOS(python.org 파이썬)에서 'CERTIFICATE_VERIFY_FAILED'가 나면:
  - 근본 해결: 터미널에서  /Applications/Python\ 3.11/Install\ Certificates.command  한 번 실행
  - 또는 아래 SSL 우회 코드(이미 포함됨)가 certifi 번들로 검증하게 해줍니다.
"""
# --- macOS python.org SSL 인증서 우회 (urllib/pandas.read_csv용) ---------
import os, ssl
# ── SSL 검증 우회 (회사/기관 네트워크의 self-signed 프록시 대응) ──────────
# 공개 주가 시세만 받으므로 기본적으로 검증을 끈다(어떤 실행 방식이든 무조건 적용).
# 보안 검증을 유지하려면  SECURE_SSL=1  환경변수로 실행.
if os.environ.get("SECURE_SSL", "0") != "1":
    try:
        import requests, urllib3
        urllib3.disable_warnings()
        _o_req = requests.Session.request
        def _req(self, *a, **k):
            k.setdefault("verify", False); return _o_req(self, *a, **k)
        requests.Session.request = _req
        _o_get = requests.get
        def _get(*a, **k):
            k.setdefault("verify", False); return _o_get(*a, **k)
        requests.get = _get
    except ImportError:
        pass
    ssl._create_default_https_context = ssl._create_unverified_context
    print("[알림] SSL 인증서 검증을 끄고 실행합니다 (공개 시세 수집용).")
else:
    try:
        import certifi
        ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        pass

import time
import pandas as pd
import FinanceDataReader as fdr
from tqdm import tqdm

# ----------------------------------------------------------------------
# 설정 — 필요에 맞게 조정
# ----------------------------------------------------------------------
MARKET = "KOSPI"          # "KOSPI" 또는 "KOSDAQ"
START = "2019-01-01"      # 수집 시작일
END = None                # None이면 오늘까지
TOP_N = 200               # 시가총액 상위 N종목만 (전체는 느리고 잡주가 많음)
PAUSE = 0.15              # 요청 간 대기(초) — 과도한 요청 방지


def main():
    # 1) 종목 리스트 + 섹터(업종) --------------------------------------
    listing = fdr.StockListing(MARKET)
    # 컬럼명은 버전에 따라 다를 수 있어 유연하게 매핑
    cols = {c.lower(): c for c in listing.columns}
    code_col = cols.get("code") or cols.get("symbol") or "Code"
    name_col = cols.get("name") or "Name"
    sector_col = (cols.get("sector") or cols.get("industry")
                  or cols.get("업종") or None)
    cap_col = (cols.get("marcap") or cols.get("marketcap")
               or cols.get("시가총액") or None)

    df = listing.copy()
    if cap_col:                                   # 시총 상위 N만
        df = df.sort_values(cap_col, ascending=False).head(TOP_N)
    else:
        df = df.head(TOP_N)

    meta = pd.DataFrame({
        "code": df[code_col].astype(str).str.zfill(6),
        "name": df[name_col],
        "market": MARKET,
        "sector": df[sector_col] if sector_col else "N/A",
    })
    meta.to_csv("kr_meta.csv", index=False, encoding="utf-8-sig")
    print(f"[meta] {len(meta)}종목 저장 → kr_meta.csv")

    # 2) 종목별 일봉(OHLCV) 수집 → 와이드 패널 5종 --------------------
    panels = {"open": {}, "high": {}, "low": {}, "close": {}, "volume": {}}
    colmap = {"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}
    for code in tqdm(meta["code"], desc=f"{MARKET} OHLCV 수집"):
        try:
            d = fdr.DataReader(code, START, END)
            if not d.empty and "Close" in d:
                for k, col in colmap.items():
                    if col in d:
                        panels[k][code] = d[col]
        except Exception as e:
            print(f"  skip {code}: {e}")
        time.sleep(PAUSE)

    if not panels["close"]:
        print("\n수집된 종목이 0개입니다. 위 로그에 SSL/인증서 오류가 있으면 네트워크(회사 방화벽) 문제입니다.")
        print(" → 개인 네트워크에서 재시도하거나,  SECURE_SSL 없이(기본) 검증 우회로 실행하세요.")
        return

    # 종가는 kr_prices.csv (기존 호환), 나머지는 kr_open/high/low/volume.csv
    files = {"close": "kr_prices.csv", "open": "kr_open.csv",
             "high": "kr_high.csv", "low": "kr_low.csv", "volume": "kr_volume.csv"}
    for k, fname in files.items():
        if not panels[k]:
            continue
        panel = pd.DataFrame(panels[k]); panel.index.name = "date"
        panel.to_csv(fname, encoding="utf-8-sig")
    cp = pd.DataFrame(panels["close"])
    print(f"[prices] {cp.shape[0]}일 x {cp.shape[1]}종목 저장 → kr_prices.csv(+open/high/low/volume)")
    print(f"기간: {cp.index.min().date()} ~ {cp.index.max().date()}")


if __name__ == "__main__":
    main()