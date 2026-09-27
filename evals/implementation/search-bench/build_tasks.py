#!/usr/bin/env python3
"""Answer keys, checkouts and a local BM25 index for the code-search benchmark.

A task is a repository at a fixed commit plus an answer key: the declarations a
real change edited (SWE-bench Verified gold patches, as packaged with
declaration names by moatless-tools) and the existing tests next to them. Any
search tool is asked a short question about that code and scored on whether it
returns those declarations, and on how much else it returns.

  build_tasks.py build                     # writes evals/runs/search-bench/tasks.jsonl
  build_tasks.py checkout ID DEST          # materialize a task's repository
  build_tasks.py bm25 DIR QUERY [--k 30]   # rank a checkout's declarations, JSON on stdout
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
import math
import re
import subprocess
import sys
import tarfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "evals/runs/search-bench"
DATASET_REPO = "https://github.com/aorwall/moatless-tools.git"
DATASET_COMMIT = "011ead57a5c81664e9c45e07e1f50b17e695cc63"
DATASET_PATH = "moatless/evaluation/swebench_verified_all_evaluations.json"
# Keeps the sample spread across projects instead of mostly Django.
PER_REPO_CAP = {"django/django": 6, "sympy/sympy": 4}
DEFAULT_CAP = 4
MAX_BLOCK_BYTES = 6000
MAX_FILE_BYTES = 1_000_000
UNRESOLVABLE = re.compile(r"^(imports|docstring|impl(:\d+)?)$")
STOP = set(
    "the a an and or of to in is it for on with as be this that are was by not from at if "
    "but when we can should would will have has use using get set self none true false "
    "return def class import where which what how does do".split()
)


def run(args: list[str], cwd: Path | None = None, data: bytes | None = None) -> bytes:
    return subprocess.run(args, cwd=cwd, input=data, check=True, capture_output=True).stdout


def literal(value):
    return ast.literal_eval(value) if isinstance(value, str) else value


def load_dataset() -> list[dict]:
    target = OUT / "cache" / "moatless-tools"
    if not (target / DATASET_PATH).exists():
        target.mkdir(parents=True, exist_ok=True)
        run(["git", "init", "-q"], target)
        run(["git", "fetch", "-q", "--depth", "1", "--filter=blob:none", DATASET_REPO, DATASET_COMMIT], target)
        run(["git", "sparse-checkout", "set", "--no-cone", DATASET_PATH], target)
        run(["git", "checkout", "-q", "FETCH_HEAD"], target)
    return json.loads((target / DATASET_PATH).read_text())


def select_tasks(dataset: list[dict], limit: int) -> list[dict]:
    eligible = []
    for task in dataset:
        spans = literal(task["expected_spans"])
        names = [name for names in spans.values() for name in names]
        if not all(path.endswith(".py") for path in spans) or not 1 <= len(names) <= 3:
            continue
        if all(UNRESOLVABLE.match(name) for name in names):
            continue
        eligible.append(task)
    eligible.sort(key=lambda task: hashlib.sha256(task["instance_id"].encode()).hexdigest())
    chosen, per_repo = [], Counter()
    for task in eligible:
        if per_repo[task["repo"]] < PER_REPO_CAP.get(task["repo"], DEFAULT_CAP):
            per_repo[task["repo"]] += 1
            chosen.append(task)
    return chosen[:limit]


def object_store(repo: str, commit: str) -> Path:
    """A shared shallow object store per repository; each task fetches only its commit."""
    git = OUT / "cache" / "repos" / (repo.replace("/", "__") + ".git")
    if not git.exists():
        git.mkdir(parents=True)
        run(["git", "init", "-q", "--bare"], git)
    try:
        run(["git", "cat-file", "-e", f"{commit}^{{tree}}"], git)
    except subprocess.CalledProcessError:
        run(["git", "fetch", "-q", "--depth", "1", f"https://github.com/{repo}.git", commit], git)
    return git


def python_files(git: Path, commit: str) -> dict[str, str]:
    listing = run(["git", "ls-tree", "-r", "-l", commit], git).decode()
    wanted = []
    for line in listing.splitlines():
        meta, path = line.split("\t", 1)
        _, kind, sha, size = meta.split()
        if kind == "blob" and path.endswith(".py") and size != "-" and int(size) <= MAX_FILE_BYTES:
            wanted.append((path, sha))
    raw = run(["git", "cat-file", "--batch"], git, "\n".join(sha for _, sha in wanted).encode() + b"\n")
    files, offset = {}, 0
    for path, _ in wanted:
        header_end = raw.index(b"\n", offset)
        size = int(raw[offset:header_end].split()[2])
        body = raw[header_end + 1 : header_end + 1 + size]
        offset = header_end + 1 + size + 1
        try:
            files[path] = body.decode("utf-8")
        except UnicodeDecodeError:
            pass
    return files


def declarations(path: str, source: str) -> list[dict]:
    """Functions, methods, and class bodies up to their first method, named like moatless spans."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return []
    lines = source.split("\n")
    blocks = []

    def start_of(node) -> int:
        return min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])

    def emit(name: str, start: int, end: int):
        text = "\n".join(lines[start - 1 : end])
        if len(text.encode()) > MAX_BLOCK_BYTES:
            text = text.encode()[:MAX_BLOCK_BYTES].decode("utf-8", "ignore").rsplit("\n", 1)[0]
        blocks.append({"path": path, "name": name, "start_line": start, "end_line": end, "source": text})

    def visit(body, prefix: str):
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                emit(prefix + node.name, start_of(node), node.end_lineno)
            elif isinstance(node, ast.ClassDef):
                members = [n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
                header_end = start_of(members[0]) - 1 if members else node.end_lineno
                while header_end > node.lineno and not lines[header_end - 1].strip():
                    header_end -= 1
                emit(prefix + node.name, start_of(node), header_end)
                visit(node.body, prefix + node.name + ".")

    visit(tree.body, "")
    return blocks


def build(limit: int):
    OUT.mkdir(parents=True, exist_ok=True)
    tasks = select_tasks(load_dataset(), limit)
    records = []
    for n, task in enumerate(tasks, 1):
        print(f"[{n}/{len(tasks)}] {task['instance_id']}", file=sys.stderr, flush=True)
        git = object_store(task["repo"], task["base_commit"])
        files = python_files(git, task["base_commit"])
        index = {(b["path"], b["name"]): b for path in files for b in declarations(path, files[path])}
        gold, unresolved = [], []
        for label, field in (("edit", "expected_spans"), ("test", "test_file_spans")):
            for path, names in literal(task[field]).items():
                for name in names:
                    block = index.get((path, name))
                    if block is None:
                        if not UNRESOLVABLE.match(name):
                            unresolved.append(f"{label}:{path}:{name}")
                    elif not any(g["path"] == path and g["name"] == name for g in gold):
                        gold.append({k: block[k] for k in ("path", "name", "start_line", "end_line")} | {"label": label})
        if not any(g["label"] == "edit" for g in gold):
            continue
        records.append(
            {
                "instance_id": task["instance_id"],
                "repo": task["repo"],
                "base_commit": task["base_commit"],
                "problem_statement": task["problem_statement"],
                "python_files": len(files),
                "gold": gold,
                "unresolved": unresolved,
            }
        )
    with open(OUT / "tasks.jsonl", "w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    labels = Counter(g["label"] for r in records for g in r["gold"])
    print(f"tasks: {len(records)}, gold declarations: {dict(labels)}, repos: {dict(Counter(r['repo'] for r in records))}")


def checkout(instance_id: str, dest: Path):
    task = next(json.loads(l) for l in open(OUT / "tasks.jsonl") if json.loads(l)["instance_id"] == instance_id)
    git = object_store(task["repo"], task["base_commit"])
    dest.mkdir(parents=True, exist_ok=True)
    archive = run(["git", "archive", "--format=tar", task["base_commit"]], git)
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(dest, filter="data")


def tokens(text: str) -> list[str]:
    out = []
    for word in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text):
        parts = [p for p in re.split(r"_|(?<=[a-z0-9])(?=[A-Z])", word) if p]
        for part in {word, *parts}:
            part = part.lower()
            if len(part) > 1 and part not in STOP:
                out.append(part)
    return out


def bm25(root: Path, query: str, k: int, k1=1.2, b=0.75) -> list[dict]:
    cache = root.parent / (root.name + ".index.json")
    if cache.exists():
        blocks = json.loads(cache.read_text())
    else:
        blocks = []
        for file in sorted(root.rglob("*.py")):
            if file.stat().st_size <= MAX_FILE_BYTES and file.is_file():
                try:
                    blocks += declarations(str(file.relative_to(root)), file.read_text("utf-8"))
                except UnicodeDecodeError:
                    pass
        cache.write_text(json.dumps(blocks))
    docs = [Counter(tokens(f"{blk['path']} {blk['name']} {blk['source']}")) for blk in blocks]
    lengths = [sum(doc.values()) for doc in docs]
    average = sum(lengths) / max(1, len(lengths))
    frequency = Counter(term for doc in docs for term in doc)
    terms = set(tokens(query))
    scores = []
    for doc, length in zip(docs, lengths):
        score = 0.0
        for term in terms:
            tf = doc.get(term)
            if tf:
                idf = math.log(1 + (len(docs) - frequency[term] + 0.5) / (frequency[term] + 0.5))
                score += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * length / average))
        scores.append(score)
    order = sorted(range(len(blocks)), key=lambda i: (-scores[i], i))[:k]
    return [blocks[i] | {"bm25": round(scores[i], 3)} for i in order]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("build").add_argument("--limit", type=int, default=44)
    co = sub.add_parser("checkout")
    co.add_argument("instance_id")
    co.add_argument("dest", type=Path)
    search = sub.add_parser("bm25")
    search.add_argument("root", type=Path)
    search.add_argument("query")
    search.add_argument("--k", type=int, default=30)
    args = parser.parse_args()
    if args.command == "build":
        build(args.limit)
    elif args.command == "checkout":
        checkout(args.instance_id, args.dest)
    else:
        json.dump(bm25(args.root, args.query, args.k), sys.stdout)


if __name__ == "__main__":
    main()
