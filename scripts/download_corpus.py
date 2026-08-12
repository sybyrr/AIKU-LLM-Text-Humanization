#!/usr/bin/env python3
"""모두의 말뭉치(국립국어원) 정식 승인본 내려받기.

국립국어원이 제공한 예제는 API 가 돌려준 URL 을 웹브라우저로 여는 형태인데,
이 환경에는 브라우저가 없으므로 그 URL 을 직접 받아 저장하도록 바꿨습니다.

    GET https://kli.korean.go.kr/restapi/v1/corpus/download?keyVal=<KEY>
      → 응답 본문이 실제 내려받기 URL (문자열)

API 키는 코드에 넣지 말고 환경변수나 파일로 주세요. 키가 곧 계정 권한입니다.

    export KLI_API_KEY="..."                 # 또는
    echo "..." > /workspace/.kli_api_key     # 이 파일은 커밋하지 말 것

사용:
    python download_corpus.py                       # 기본 경로에 저장
    python download_corpus.py --out /path/dir
    python download_corpus.py --url-only            # 내려받지 않고 URL 만 출력
"""
import argparse, os, pathlib, re, sys, time, urllib.parse, urllib.request

API = "https://kli.korean.go.kr/restapi/v1/corpus/download"

# 이 파일은 컨테이너 안(/workspace/scripts)과 호스트 양쪽에서 실행됩니다.
# 경로를 하드코딩하면 한쪽에서만 동작하므로, 스크립트 위치를 기준으로 잡습니다.
ROOT = pathlib.Path(__file__).resolve().parent.parent      # …/scripts/.. = 저장소 루트
DEFAULT_OUT = ROOT / "corpus" / "raw"
KEY_FILES = [ROOT / ".kli_api_key", pathlib.Path.cwd() / ".kli_api_key",
             pathlib.Path.home() / ".kli_api_key"]


def read_key(cli_key):
    if cli_key:
        return cli_key.strip()
    if os.environ.get("KLI_API_KEY"):
        return os.environ["KLI_API_KEY"].strip()
    for kf in KEY_FILES:
        if kf.exists():
            k = kf.read_text().strip()
            if k:
                print(f"키 파일: {kf}")
                return k
    tried = "\n".join(f"    {k}" for k in KEY_FILES)
    sys.exit(
        f"API 키를 찾지 못했습니다. 아래 경로를 확인했습니다:\n{tried}\n\n"
        f"셋 중 하나로 주세요:\n"
        f"  export KLI_API_KEY='...'\n"
        f"  echo '...' > {KEY_FILES[0]}\n"
        f"  python {sys.argv[0]} --key '...'")


def mask(text, key):
    """출력물에서 키를 가린다. 응답 URL 에 keyVal 이 그대로 실려 오므로 필수."""
    out = text.replace(key, f"…{key[-4:]}")
    return re.sub(r"(keyVal=)[^&\s]+", r"\1…", out)


def get_download_url(key, timeout=60):
    url = f"{API}?{urllib.parse.urlencode({'keyVal': key})}"
    req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        if r.status != 200:
            raise RuntimeError(f"API 응답 {r.status}")
        body = r.read().decode("utf-8", "replace").strip()
    if not body.startswith("http"):
        # 국립국어원은 오류도 200 으로 본문에 실어 보내는 경우가 있습니다
        raise RuntimeError(f"URL 이 아닌 응답: {mask(body[:300], key)}")
    return body


def filename_from(url, resp):
    cd = resp.headers.get("Content-Disposition", "")
    if "filename=" in cd:
        return urllib.parse.unquote(cd.split("filename=")[-1].strip('"; '))
    name = pathlib.Path(urllib.parse.urlparse(url).path).name
    return name or "corpus.zip"


def download(url, outdir, timeout=120):
    outdir = pathlib.Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        dest = outdir / filename_from(url, r)
        total = int(r.headers.get("Content-Length") or 0)
        print(f"저장: {dest}" + (f"  ({total/1e9:.1f} GB)" if total else ""))
        done, t0, tick = 0, time.time(), 0.0
        with open(dest, "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
                done += len(chunk)
                if time.time() - tick > 2:
                    tick = time.time()
                    el = tick - t0
                    pct = f"{done/total*100:5.1f}%" if total else "     "
                    print(f"\r  {pct} {done/1e9:6.2f} GB  {done/1e6/max(el,1):6.1f} MB/s",
                          end="", flush=True)
        print(f"\r  완료 {done/1e9:.2f} GB  {time.time()-t0:.0f}초" + " " * 20)
    return dest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", help="API 키 (미지정 시 env KLI_API_KEY 또는 키 파일)")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--url-only", action="store_true", help="URL 만 출력하고 종료")
    ap.add_argument("--show-key", action="store_true",
                    help="출력에서 키를 가리지 않음 (기본은 가림)")
    args = ap.parse_args()

    key = read_key(args.key)
    print(f"API 호출: {API}  (키 …{key[-4:]})")
    try:
        url = get_download_url(key)
    except Exception as e:
        sys.exit(f"API 호출 실패: {e}")

    if args.url_only:
        # 응답 URL 에는 keyVal 이 그대로 들어 있다. 기본은 가리고,
        # 정말 원문이 필요할 때만 --show-key 로 꺼낸다.
        print(url if args.show_key else mask(url, key))
        if not args.show_key:
            print("  (keyVal 가림. 원문은 --show-key)")
        return

    try:
        dest = download(url, args.out)
    except Exception as e:
        sys.exit(f"내려받기 실패: {mask(str(e), key)}\n  URL: {mask(url, key)}")

    # 압축 파일이면 안내만. 자동 해제는 하지 않습니다(용량·덮어쓰기 위험).
    if dest.suffix.lower() in (".zip", ".gz", ".tar"):
        print(f"\n압축 해제:\n  cd {dest.parent} && unzip -o {dest.name}")
        print("  (컨테이너에 unzip 이 없으면: "
              f"python -m zipfile -e {dest.name} .)")


if __name__ == "__main__":
    main()
