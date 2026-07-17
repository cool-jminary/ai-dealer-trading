"""
상장주식수 수집 (본인 PC 1회 실행) — 시총 상위 바스켓 전략용

대형주는 상장주식수가 거의 안 변하므로 현재 시점 값 한 번이면 충분하다.
이걸로 시점별 시가총액 ≈ 주가 × 상장주식수 로 근사해 시총 순위를 매긴다.

준비:
  pip install finance-datareader pandas certifi
  같은 폴더에 kr_meta.csv
실행:
  python fetch_shares.py

산출물:
  kr_shares.csv  — code, name, shares(상장주식수), marcap(참고: 현재 시총)

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

import pandas as pd
import FinanceDataReader as fdr


def main():
    meta = pd.read_csv("kr_meta.csv", dtype={"code": str})
    meta["code"] = meta["code"].str.zfill(6)
    want = set(meta["code"])

    # KOSPI 전체 상장정보 (Stocks=상장주식수, Marcap=시가총액 포함)
    listing = fdr.StockListing("KOSPI")
    cols = {c.lower(): c for c in listing.columns}
    code_col = cols.get("code") or cols.get("symbol") or "Code"
    name_col = cols.get("name") or "Name"
    shares_col = cols.get("stocks") or cols.get("shares") or cols.get("상장주식수")
    cap_col = cols.get("marcap") or cols.get("marketcap") or cols.get("시가총액")

    df = listing.copy()
    df[code_col] = df[code_col].astype(str).str.zfill(6)
    df = df[df[code_col].isin(want)]

    out = pd.DataFrame({"code": df[code_col], "name": df[name_col]})
    if shares_col:
        out["shares"] = pd.to_numeric(df[shares_col], errors="coerce")
    else:
        print("경고: 상장주식수 컬럼을 못 찾음. StockListing 컬럼:", list(listing.columns))
    if cap_col:
        out["marcap"] = pd.to_numeric(df[cap_col], errors="coerce")

    out = out.dropna(subset=["shares"]) if "shares" in out else out
    out.to_csv("kr_shares.csv", index=False, encoding="utf-8-sig")
    print(f"저장 완료 → kr_shares.csv ({len(out)}종목)")
    print("컬럼:", list(out.columns))
    if "marcap" in out:
        top = out.sort_values("marcap", ascending=False).head(10)
        print("\n현재 시총 상위 10:")
        print(top[["code", "name", "marcap"]].to_string(index=False))


if __name__ == "__main__":
    main()
