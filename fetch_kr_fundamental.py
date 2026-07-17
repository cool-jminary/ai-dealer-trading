"""
한국 주식 밸류/재무 지표 수집 (본인 PC 실행) — 안정화 버전

변경점:
  - 영업일을 KRX에 묻지 않고 기존 kr_prices.csv의 실제 거래일에서 뽑음 (IndexError 방지)
  - 날짜별 조회에 재시도+대기 → KRX의 일시적 빈 응답을 넘김
  - 실패한 날짜는 건너뛰고 계속, 부분 결과라도 저장

준비:
  pip install --upgrade pykrx            # ★ 오래된 pykrx가 원인인 경우가 많음
  pip install pandas tqdm certifi
  같은 폴더에 kr_prices.csv, kr_meta.csv 필요

산출물:
  kr_fundamental.csv  — date, code, BPS, PER, PBR, EPS, DIV, DPS, ROE(근사)

※ macOS SSL 오류 시: /Applications/Python\ 3.11/Install\ Certificates.command 실행
"""
import ssl, time
try:
    import certifi
    ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())
except ImportError:
    pass

import pandas as pd
from tqdm import tqdm
from pykrx import stock

FREQ_DAYS = 21          # 며칠 간격 스냅샷 (21≈월 1회)
RETRY = 3               # KRX 빈 응답 시 재시도 횟수
PAUSE = 0.6             # 요청 간 대기(초) — 너무 빠르면 KRX가 막음


def snapshot_dates(prices_csv="kr_prices.csv"):
    close = pd.read_csv(prices_csv, index_col=0, parse_dates=True)
    return [d.strftime("%Y%m%d") for d in close.index[::FREQ_DAYS]]


def fundamental_on(date):
    """특정일 KOSPI 전 종목 PER/PBR/EPS/BPS — 재시도 포함"""
    for k in range(RETRY):
        try:
            df = stock.get_market_fundamental_by_ticker(date, market="KOSPI")
            if df is not None and not df.empty:
                return df
        except Exception as e:
            if k == RETRY - 1:
                print(f"  · {date} 실패: {e}")
        time.sleep(PAUSE * (k + 1))
    return None


def main():
    codes = set(pd.read_csv("kr_meta.csv", dtype={"code": str})["code"].str.zfill(6))
    dates = snapshot_dates()
    if not dates:
        print("거래일을 찾지 못했습니다. kr_prices.csv를 확인하세요.")
        return
    print(f"종목 {len(codes)}개 · 스냅샷 {len(dates)}개 ({dates[0]}~{dates[-1]})")

    rows, ok = [], 0
    for d in tqdm(dates, desc="펀더멘털 수집"):
        df = fundamental_on(d)
        if df is None:
            continue
        df = df.reset_index().rename(columns={"티커": "code"})
        df["code"] = df["code"].astype(str).str.zfill(6)
        df = df[df["code"].isin(codes)]
        df.insert(0, "date", pd.to_datetime(d))
        rows.append(df); ok += 1
        time.sleep(PAUSE)

    if not rows:
        print("\n수집 실패 — pykrx 업그레이드(pip install --upgrade pykrx) 후 재시도하거나,")
        print("잠시 뒤(레이트리밋) 또는 다른 네트워크에서 다시 실행하세요.")
        return

    out = pd.concat(rows, ignore_index=True)
    if {"EPS", "BPS"}.issubset(out.columns):
        out["ROE"] = (out["EPS"] / out["BPS"]).replace([float("inf"), -float("inf")], pd.NA)
    out.to_csv("kr_fundamental.csv", index=False, encoding="utf-8-sig")
    print(f"\n저장 완료 → kr_fundamental.csv ({ok}/{len(dates)} 스냅샷, {len(out)}행)")
    print("컬럼:", list(out.columns))


if __name__ == "__main__":
    main()
