"""
2025년 재무만 받아 기존 kr_fundamental.csv 에 이어붙이기 (append)

- 전체 재수집 없이 새 연도만 추가한다.
- 기존 파일에 이미 있는 (연도, 종목)은 건너뛰어 중복을 막는다.
- opendart_fetch.py 의 검증된 함수를 그대로 재사용한다.

준비:
  export DART_API_KEY=발급받은키
  같은 폴더에 opendart_fetch.py, kr_meta.csv, (있으면) kr_fundamental.csv
실행:
  python append_year.py           # 기본 2025년 추가
"""
import os
import pandas as pd

# opendart_fetch.py 의 함수·설정 재사용
import opendart_fetch as od

ADD_YEARS = [2025]                       # 추가할 연도
OUT = "kr_fundamental.csv"


def main():
    if od.API_KEY.startswith("여기에"):
        print("API 키를 먼저 설정하세요: export DART_API_KEY=발급키")
        return

    # 기존 파일 로드 (없으면 빈 것으로 시작)
    if os.path.exists(OUT):
        old = pd.read_csv(OUT, dtype={"code": str})
        old["code"] = old["code"].str.zfill(6)
        have = set(zip(old["fiscal_year"], old["code"]))
        print(f"기존 {len(old)}행 로드 · 연도 {sorted(old['fiscal_year'].unique())}")
    else:
        old = pd.DataFrame()
        have = set()
        print("기존 kr_fundamental.csv 없음 → 새로 생성")

    codes = set(pd.read_csv("kr_meta.csv", dtype={"code": str})["code"].str.zfill(6))
    print("고유번호 매핑 다운로드 중…")
    cmap = od.corp_map(codes)

    from tqdm import tqdm
    import time
    rows = []
    for code, corp in tqdm(cmap.items(), desc="2025 재무 수집"):
        for y in ADD_YEARS:
            if (y, code) in have:            # 이미 있으면 스킵 (중복 방지)
                continue
            ni, eq = od.get_financials(corp, y)
            if ni is None or eq is None:
                continue
            sh = od.get_shares(corp, y)
            if not sh:
                continue
            rows.append({
                "fiscal_year": y,
                "available_from": (pd.Timestamp(y, 12, 31) + pd.Timedelta(days=od.LAG_DAYS)).date(),
                "code": code, "net_income": ni, "equity": eq, "shares": sh,
                "EPS": ni / sh, "BPS": eq / sh, "ROE": ni / eq if eq else None,
            })
            time.sleep(od.PAUSE)

    if not rows:
        print("추가된 데이터가 없습니다 (이미 최신이거나 2025 사업보고서 미공시).")
        return

    new = pd.DataFrame(rows)
    merged = pd.concat([old, new], ignore_index=True)
    merged = merged.sort_values(["code", "fiscal_year"]).reset_index(drop=True)
    merged.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"\n{len(new)}행 추가 → {OUT} (총 {len(merged)}행, 연도 {sorted(merged['fiscal_year'].unique())})")
    print(new.head(6).to_string(index=False))


if __name__ == "__main__":
    main()
