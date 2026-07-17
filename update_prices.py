"""
시세 오늘자 갱신 — 기존 kr_prices.csv 를 마지막 날짜 다음날부터 오늘까지 이어붙이기

전체 재수집 없이 '없는 최근 구간'만 받아 붙인다(증분 업데이트). 거래량 패널도 함께 갱신.
발표 당일 서버 켤 때 자동 호출하면 오늘 시점으로 선정·주문이 돈다.

★ 거래량(kr_volume.csv)이 아직 없다면 이 증분 스크립트는 '갱신 시점 이후' 거래량만 채운다.
  과거 전체 거래량이 필요하면 먼저 fetch_kr_data.py 를 한 번 다시 돌려 전체 재수집할 것.

준비:
  pip install finance-datareader pandas tqdm certifi
  같은 폴더에 kr_prices.csv, kr_meta.csv
실행:
  python update_prices.py          # 오늘까지 갱신

※ macOS SSL 오류 시: /Applications/Python\ 3.11/Install\ Certificates.command 실행
"""
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

PRICES = "kr_prices.csv"
META = "kr_meta.csv"
PAUSE = 0.1
FILES = {"close": "kr_prices.csv", "open": "kr_open.csv",
         "high": "kr_high.csv", "low": "kr_low.csv", "volume": "kr_volume.csv"}
COLMAP = {"close": "Close", "open": "Open", "high": "High", "low": "Low", "volume": "Volume"}


def _merge_save(fname, base_close, add_df):
    add_df.index.name = "date"
    add_df = add_df.reindex(columns=base_close.columns)
    if os.path.exists(fname):
        old = pd.read_csv(fname, index_col=0, parse_dates=True)
        old.columns = [str(c).zfill(6) for c in old.columns]
    else:
        old = pd.DataFrame(columns=base_close.columns)
    merged = pd.concat([old, add_df])
    merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    merged.to_csv(fname, encoding="utf-8-sig")
    return merged


def main():
    close = pd.read_csv(PRICES, index_col=0, parse_dates=True)
    close.columns = [str(c).zfill(6) for c in close.columns]
    last = close.index.max()
    start = (last + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    today = pd.Timestamp.today().normalize()

    if last >= today:
        print(f"이미 최신입니다 (마지막 {last.date()}).")
        return
    print(f"기존 마지막: {last.date()} → {today.date()} 까지 OHLC 증분 수집")

    codes = pd.read_csv(META, dtype={"code": str})["code"].str.zfill(6).tolist()
    codes = [c for c in codes if c in close.columns] or list(close.columns)

    add = {"open": {}, "high": {}, "low": {}, "close": {}, "volume": {}}
    for code in tqdm(codes, desc="OHLC 증분"):
        try:
            d = fdr.DataReader(code, start)
            if not d.empty and "Close" in d:
                for k, col in COLMAP.items():
                    if col in d:
                        add[k][code] = d[col]
        except Exception as e:
            print(f"  skip {code}: {e}")
        time.sleep(PAUSE)

    if not add["close"]:
        print("추가된 거래일이 없습니다 (휴장 또는 이미 최신).")
        return

    merged_close = None
    for k, fname in FILES.items():
        if not add[k]:
            continue
        m = _merge_save(fname, close, pd.DataFrame(add[k]))
        if k == "close":
            merged_close = m
    n = len(pd.DataFrame(add["close"]))
    print(f"\n{n}거래일 추가 → kr_prices.csv(+open/high/low) "
          f"(총 {merged_close.shape[0]}일, {merged_close.index.min().date()}~{merged_close.index.max().date()})")


if __name__ == "__main__":
    main()