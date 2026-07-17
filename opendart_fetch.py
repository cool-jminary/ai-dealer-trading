"""
OpenDART(금감원 전자공시) 재무데이터 수집 → 밸류 지표  (본인 PC 실행)

KRX 웹 스크래핑과 달리 공식 오픈 API라 안정적이다.
회사별·연도별 재무제표에서 '당기순이익·자본총계'와 '발행주식수'를 받아
EPS·BPS·ROE를 계산한다. PER/PBR은 백테스트에서 가격과 결합해 산출한다.

준비:
  1) https://opendart.fss.or.kr 에서 무료 API 키 발급
  2) pip install requests pandas tqdm
  3) 같은 폴더에 kr_meta.csv (code 컬럼) 필요
  4) 키 설정:  export DART_API_KEY=발급받은키   (또는 아래 API_KEY에 직접 입력)

산출물:
  kr_fundamental.csv  — fiscal_year, available_from, code, net_income, equity,
                        shares, EPS, BPS, ROE
  * available_from = 회계연도말 + 90일(사업보고서 공시시차). 백테스트가 이 시점부터
    해당 재무를 '알 수 있는' 것으로 취급 → look-ahead 방지.
"""
import os, io, time, zipfile
import xml.etree.ElementTree as ET
import requests
import pandas as pd
from tqdm import tqdm

API_KEY = os.environ.get("DART_API_KEY", "여기에_발급받은_키_입력")
YEARS = [2019, 2020, 2021, 2022, 2023, 2024, 2025]
REPRT = "11011"          # 사업보고서(연간)
LAG_DAYS = 90            # 공시 시차
PAUSE = 0.08
BASE = "https://opendart.fss.or.kr/api"


# ---- 검증된 파싱 함수 (test_parse.py로 단위 검증 완료) ----------------
def to_num(s):
    if s is None:
        return None
    s = str(s).replace(",", "").strip()
    if s in ("", "-"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def parse_financials(items):
    net_income, equity = None, None
    for it in items:
        aid = it.get("account_id", ""); nm = (it.get("account_nm") or "").strip()
        sj = it.get("sj_div", ""); amt = to_num(it.get("thstrm_amount"))
        if equity is None and (aid == "ifrs-full_Equity" or nm == "자본총계") and sj == "BS":
            equity = amt
        if net_income is None and (aid == "ifrs-full_ProfitLoss" or "당기순이익" in nm) and sj in ("IS", "CIS"):
            net_income = amt
    return net_income, equity


def parse_shares(items):
    for it in items:
        if "보통주" in (it.get("se") or ""):
            v = to_num(it.get("isu_stock_totqy"))
            if v:
                return v
    for it in items:
        if (it.get("se") or "").strip() in ("합계", "계"):
            v = to_num(it.get("isu_stock_totqy"))
            if v:
                return v
    return None


# ---- API 호출 --------------------------------------------------------
def corp_map(codes):
    """종목코드(6자리) → OpenDART 고유번호(8자리) 매핑"""
    r = requests.get(f"{BASE}/corpCode.xml", params={"crtfc_key": API_KEY}, timeout=30)
    zf = zipfile.ZipFile(io.BytesIO(r.content))
    root = ET.fromstring(zf.read(zf.namelist()[0]))
    m = {}
    for e in root.iter("list"):
        sc = (e.findtext("stock_code") or "").strip()
        cc = (e.findtext("corp_code") or "").strip()
        if sc and sc.zfill(6) in codes:
            m[sc.zfill(6)] = cc
    return m


def _get(endpoint, params):
    for k in range(3):
        try:
            j = requests.get(f"{BASE}/{endpoint}", params=params, timeout=20).json()
            if j.get("status") == "000":
                return j.get("list", [])
            if j.get("status") == "013":     # 데이터 없음
                return []
        except Exception:
            pass
        time.sleep(0.3 * (k + 1))
    return []


def get_financials(corp, year):
    for fs in ("CFS", "OFS"):                # 연결 우선, 없으면 별도
        items = _get("fnlttSinglAcntAll.json",
                     {"crtfc_key": API_KEY, "corp_code": corp, "bsns_year": str(year),
                      "reprt_code": REPRT, "fs_div": fs})
        ni, eq = parse_financials(items)
        if ni is not None and eq is not None:
            return ni, eq
    return None, None


def get_shares(corp, year):
    items = _get("stockTotqySttus.json",
                 {"crtfc_key": API_KEY, "corp_code": corp, "bsns_year": str(year), "reprt_code": REPRT})
    return parse_shares(items)


def main():
    if API_KEY.startswith("여기에"):
        print("먼저 API 키를 설정하세요: export DART_API_KEY=발급키  (또는 코드 상단 API_KEY 수정)")
        return
    codes = set(pd.read_csv("kr_meta.csv", dtype={"code": str})["code"].str.zfill(6))
    print("고유번호 매핑 다운로드 중…")
    cmap = corp_map(codes)
    print(f"매핑 {len(cmap)}/{len(codes)}종목")

    rows = []
    for code, corp in tqdm(cmap.items(), desc="재무 수집"):
        for y in YEARS:
            ni, eq = get_financials(corp, y)
            if ni is None or eq is None:
                continue
            sh = get_shares(corp, y)
            if not sh:
                continue
            rows.append({
                "fiscal_year": y,
                "available_from": (pd.Timestamp(y, 12, 31) + pd.Timedelta(days=LAG_DAYS)).date(),
                "code": code, "net_income": ni, "equity": eq, "shares": sh,
                "EPS": ni / sh, "BPS": eq / sh, "ROE": ni / eq if eq else None,
            })
            time.sleep(PAUSE)

    if not rows:
        print("수집 실패 — API 키/네트워크를 확인하세요.")
        return
    out = pd.DataFrame(rows)
    out.to_csv("kr_fundamental.csv", index=False, encoding="utf-8-sig")
    print(f"\n저장 완료 → kr_fundamental.csv ({out['code'].nunique()}종목 x {len(YEARS)}년, {len(out)}행)")
    print(out.head(6).to_string(index=False))


if __name__ == "__main__":
    main()
