"""
LLM 클라이언트 (OpenAI ChatGPT 4o)

- M/O 규정 판단, 선정 근거 설명에서 공통 사용
- 환경변수 OPENAI_API_KEY 필요. 없으면 available()=False → 호출부가 규칙 기반으로 폴백
- 회사/기관 SSL 가로채기 대비: OPENAI_INSECURE_SSL=1 이면 인증서 검증 우회
- 모델: 기본 gpt-4o (환경변수 OPENAI_MODEL 로 변경 가능)

pip install openai
"""
import os
import json

MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o")
_client = None
_tried = False


def _make_client():
    """OpenAI 클라이언트 1회 생성. 실패하면 None."""
    global _client, _tried
    if _tried:
        return _client
    _tried = True
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        return None
    try:
        from openai import OpenAI
        if os.environ.get("OPENAI_INSECURE_SSL") == "1":
            import httpx
            _client = OpenAI(api_key=key, http_client=httpx.Client(verify=False))
        else:
            _client = OpenAI(api_key=key)
    except Exception as e:
        print(f"[llm] 클라이언트 생성 실패 → 규칙 기반 폴백: {e}")
        _client = None
    return _client


def available() -> bool:
    return _make_client() is not None


def chat(system: str, user: str, temperature: float = 0.2,
         max_tokens: int = 500, want_json: bool = False):
    """LLM 호출. 실패/미설정이면 None 반환(호출부가 폴백)."""
    client = _make_client()
    if client is None:
        return None
    try:
        kwargs = dict(model=MODEL, temperature=temperature, max_tokens=max_tokens,
                      messages=[{"role": "system", "content": system},
                                {"role": "user", "content": user}])
        if want_json:
            kwargs["response_format"] = {"type": "json_object"}
        resp = client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"[llm] 호출 실패 → 폴백: {e}")
        return None


def chat_json(system: str, user: str, **kw):
    """JSON 응답 전용. 파싱 실패 시 None."""
    txt = chat(system, user, want_json=True, **kw)
    if txt is None:
        return None
    try:
        return json.loads(txt)
    except Exception:
        # ```json ... ``` 방어
        t = txt.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            return json.loads(t)
        except Exception as e:
            print(f"[llm] JSON 파싱 실패 → 폴백: {e}")
            return None
