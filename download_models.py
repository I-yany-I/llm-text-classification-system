"""Download embedding and Cross-Encoder weights used by the campus RAG pipeline.

Tries ModelScope HTTP (no extra SDK) first, then Hugging Face Hub with a
single worker so large files can resume. Local snapshots go under ``models/``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from src.campus_kb_rag.config import LOCAL_MODELS_DIR, resolve_model_source


DEFAULT_MODELS = [
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
]

MODELSCOPE_API = "https://www.modelscope.cn/api/v1/models/{repo}/repo/files"
MODELSCOPE_FILE = "https://www.modelscope.cn/models/{repo}/resolve/master/{path}"

SKIP_PREFIXES = ("onnx/", "openvino/")
SKIP_FILES = {
    "tf_model.h5",
    "flax_model.msgpack",
    "rust_model.ot",
    ".gitattributes",
}


def _local_dir(repo_id: str) -> Path:
    return LOCAL_MODELS_DIR / repo_id


def _clear_proxies() -> None:
    for key in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ):
        os.environ.pop(key, None)


def _http_json(url: str, timeout: int = 60) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "nju-campus-kb-rag/download"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _download_file(url: str, dest: Path, timeout: int = 600) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    existing = tmp.stat().st_size if tmp.exists() else 0
    headers = {"User-Agent": "nju-campus-kb-rag/download"}
    if existing:
        headers["Range"] = f"bytes={existing}-"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        if existing and getattr(resp, "status", 200) == 200:
            existing = 0
        mode = "ab" if existing else "wb"
        with tmp.open(mode) as out:
            if not existing:
                out.seek(0)
                out.truncate()
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
    tmp.replace(dest)


def _modelscope_file_entries(repo_id: str) -> list[dict]:
    encoded = urllib.parse.quote(repo_id, safe="")
    payload = _http_json(MODELSCOPE_API.format(repo=encoded) + "?Revision=master&Recursive=true")
    data = payload.get("Data") or payload.get("data") or payload
    files = data.get("Files") or data.get("files") or []
    if not isinstance(files, list):
        raise RuntimeError(f"unexpected ModelScope listing for {repo_id}: {type(files)}")
    names = {
        str(item.get("Path") or item.get("path") or "")
        for item in files
        if isinstance(item, dict)
    }
    skip_bin = "model.safetensors" in names
    entries = []
    for item in files:
        if not isinstance(item, dict):
            continue
        path = str(item.get("Path") or item.get("path") or item.get("Name") or item.get("name") or "").lstrip("/")
        kind = str(item.get("Type") or item.get("type") or "blob").lower()
        if not path or kind in {"tree", "dir", "directory"}:
            continue
        if path in SKIP_FILES or any(path.startswith(prefix) for prefix in SKIP_PREFIXES):
            continue
        if skip_bin and path == "pytorch_model.bin":
            continue
        entries.append({"path": path, "size": int(item.get("Size") or item.get("size") or 0)})
    if not entries:
        raise RuntimeError(f"ModelScope returned no files for {repo_id}")
    return entries


def _download_modelscope(repo_id: str, local_dir: Path) -> Path:
    local_dir.mkdir(parents=True, exist_ok=True)
    entries = _modelscope_file_entries(repo_id)
    print(f"modelscope files {len(entries)} for {repo_id}", flush=True)
    for item in entries:
        rel = item["path"]
        dest = local_dir / rel
        if dest.exists() and dest.stat().st_size > 0:
            if item.get("size") and dest.stat().st_size == item["size"]:
                print(f"skip {rel}", flush=True)
                continue
        url = MODELSCOPE_FILE.format(repo=repo_id, path=urllib.parse.quote(rel, safe="/"))
        size_mb = item.get("size", 0) / 1e6
        print(f"get {rel} ({size_mb:.1f} MB)", flush=True)
        _download_file(url, dest)
    return local_dir


def _download_huggingface(repo_id: str, local_dir: Path) -> Path:
    from huggingface_hub import snapshot_download

    local_dir.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        max_workers=1,
        resume_download=True,
    )
    return Path(path)


def download_one(repo_id: str, source: str) -> Path:
    resolved = resolve_model_source(repo_id)
    if resolved != repo_id:
        print(f"already local: {repo_id} -> {resolved}", flush=True)
        return Path(resolved)

    local_dir = _local_dir(repo_id)
    print(f"downloading {repo_id} via {source} -> {local_dir}", flush=True)
    if source == "huggingface":
        return _download_huggingface(repo_id, local_dir)
    try:
        return _download_modelscope(repo_id, local_dir)
    except Exception as exc:
        if source == "modelscope":
            raise
        print(f"modelscope failed ({type(exc).__name__}: {exc}); falling back to huggingface", flush=True)
        return _download_huggingface(repo_id, local_dir)


def smoke_load(embed_id: str, ce_id: str) -> None:
    from sentence_transformers import CrossEncoder, SentenceTransformer

    embedder = SentenceTransformer(resolve_model_source(embed_id))
    vec = embedder.encode(["校园 VPN 怎么用"], convert_to_numpy=True)
    print("embed_ok", vec.shape, flush=True)

    ce = CrossEncoder(resolve_model_source(ce_id))
    score = ce.predict([("校园 VPN 怎么用", "南京大学 VPN 使用说明")])
    value = float(score[0] if hasattr(score, "__len__") else score)
    print("ce_ok", value, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download campus RAG encoder models.")
    parser.add_argument(
        "--source",
        choices=("auto", "modelscope", "huggingface"),
        default="auto",
        help="auto tries ModelScope HTTP first, then Hugging Face.",
    )
    parser.add_argument("--skip-smoke", action="store_true")
    args = parser.parse_args()
    _clear_proxies()

    LOCAL_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for repo_id in DEFAULT_MODELS:
        path = download_one(repo_id, args.source)
        print("cached", path, flush=True)

    if not args.skip_smoke:
        smoke_load(*DEFAULT_MODELS)
    print("MODELS_READY", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
    except urllib.error.URLError as exc:
        print(f"download failed: {exc}", file=sys.stderr)
        sys.exit(1)
