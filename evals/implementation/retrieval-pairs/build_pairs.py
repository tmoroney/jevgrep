#!/usr/bin/env python3
"""Build labeled (task, code block) pairs for measuring Jev relevance judgments.

Ground truth comes from SWE-bench Verified as packaged by moatless-tools, whose
`expected_spans` name the declarations each gold patch edits. Every task becomes
one record: the problem statement, plus a small candidate set of Python
declaration blocks read at the task's base commit:

- edit:     a declaration the gold patch changes (positive)
- test:     an existing test declaration named in `test_file_spans`
- negative: a hard negative the gold patch does not touch, tagged by origin
            (bm25 = a top lexical match for the problem statement,
             sibling = another declaration in an edited file,
             random = any other declaration in the repository)

Hard negatives come from a local BM25 index over every declaration, i.e. the
blocks a code-side prefilter would actually hand to Jev. The record also keeps
each positive's BM25 rank, which measures whether that prefilter can surface it.

Output (all under the ignored evals/runs/ tree by default):
  pairs.jsonl  one task per line
  files.jsonl  full source of every file holding a candidate, for baselines that
               reproduce jevgrep's whole-file request context
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import random
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATASET_REPO = "https://github.com/aorwall/moatless-tools.git"
DATASET_COMMIT = "011ead57a5c81664e9c45e07e1f50b17e695cc63"
DATASET_PATH = "moatless/evaluation/swebench_verified_all_evaluations.json"
# Keeps the sample spread across projects instead of mostly Django.
PER_REPO_CAP = {"django/django": 6, "sympy/sympy": 4}
DEFAULT_CAP = 4
MAX_BLOCK_BYTES = 6000
MAX_FILE_BYTES = 1_000_000
BM25_NEGATIVES = 6
SIBLING_NEGATIVES = 2
RANDOM_NEGATIVES = 1
UNRESOLVABLE = re.compile(r"^(imports|docstring|impl(:\d+)?)$")
STOP = set(
    "the a an and or of to in is it for on with as be this that are was by not from at if "
    "but when we can should would will have has use using get set self none true false "
    "return def class import none issue bug error example code expected actual".split()
)


def run(args: list[str], cwd: Path | None = None, data: bytes | None = None) -> bytes:
    return subprocess.run(args, cwd=cwd, input=data, check=True, capture_output=True).stdout


def seeded(key: str) -> random.Random:
    return random.Random(int(hashlib.sha256(key.encode()).hexdigest()[:16], 16))


def literal(value):
    return ast.literal_eval(value) if isinstance(value, str) else value


def load_dataset(cache: Path) -> list[dict]:
    target = cache / "moatless-tools"
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


class Snapshot:
    """Python sources of one repository commit, read from a shared shallow object store."""

    def __init__(self, cache: Path, repo: str, commit: str):
        self.git = cache / "repos" / (repo.replace("/", "__") + ".git")
        if not self.git.exists():
            self.git.mkdir(parents=True)
            run(["git", "init", "-q", "--bare"], self.git)
        self.commit = commit
        try:
            run(["git", "cat-file", "-e", f"{commit}^{{tree}}"], self.git)
        except subprocess.CalledProcessError:
            run(["git", "fetch", "-q", "--depth", "1", f"https://github.com/{repo}.git", commit], self.git)

    def python_files(self) -> dict[str, str]:
        listing = run(["git", "ls-tree", "-r", "-l", self.commit], self.git).decode()
        wanted = []
        for line in listing.splitlines():
            meta, path = line.split("\t", 1)
            _, kind, sha, size = meta.split()
            if kind == "blob" and path.endswith(".py") and size != "-" and int(size) <= MAX_FILE_BYTES:
                wanted.append((path, sha))
        raw = run(["git", "cat-file", "--batch"], self.git, "\n".join(sha for _, sha in wanted).encode() + b"\n")
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
        decorators = getattr(node, "decorator_list", [])
        return min([node.lineno] + [d.lineno for d in decorators])

    def emit(name: str, start: int, end: int):
        text = "\n".join(lines[start - 1 : end])
        truncated = len(text.encode()) > MAX_BLOCK_BYTES
        if truncated:
            text = text.encode()[:MAX_BLOCK_BYTES].decode("utf-8", "ignore").rsplit("\n", 1)[0]
        blocks.append(
            {"path": path, "name": name, "start_line": start, "end_line": end, "source": text, "truncated": truncated}
        )

    def visit(body, prefix: str):
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                emit(prefix + node.name, start_of(node), node.end_lineno)
            elif isinstance(node, ast.ClassDef):
                methods = [n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
                header_end = start_of(methods[0]) - 1 if methods else node.end_lineno
                while header_end > node.lineno and not lines[header_end - 1].strip():
                    header_end -= 1
                emit(prefix + node.name, start_of(node), header_end)
                visit(node.body, prefix + node.name + ".")

    visit(tree.body, "")
    return blocks


def tokens(text: str) -> list[str]:
    out = []
    for word in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text):
        parts = [p for p in re.split(r"_|(?<=[a-z0-9])(?=[A-Z])", word) if p]
        for part in {word, *parts}:
            part = part.lower()
            if len(part) > 1 and part not in STOP:
                out.append(part)
    return out


def bm25_rank(query: str, blocks: list[dict], k1=1.2, b=0.75) -> list[int]:
    docs = [Counter(tokens(f"{blk['path']} {blk['name']} {blk['source']}")) for blk in blocks]
    lengths = [sum(doc.values()) for doc in docs]
    average = sum(lengths) / max(1, len(lengths))
    frequency = Counter(term for doc in docs for term in doc)
    terms = set(tokens(query))
    count = len(docs)
    scores = []
    for doc, length in zip(docs, lengths):
        score = 0.0
        for term in terms:
            tf = doc.get(term)
            if tf:
                idf = math.log(1 + (count - frequency[term] + 0.5) / (frequency[term] + 0.5))
                score += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * length / average))
        scores.append(score)
    return sorted(range(len(blocks)), key=lambda i: (-scores[i], i))


def build_task(task: dict, cache: Path) -> tuple[dict, dict[str, str]] | None:
    snapshot = Snapshot(cache, task["repo"], task["base_commit"])
    files = snapshot.python_files()
    index = [blk for path, source in files.items() for blk in declarations(path, source)]
    by_key = {(blk["path"], blk["name"]): i for i, blk in enumerate(index)}
    labels: dict[int, str] = {}
    unresolved = []
    for kind, field in (("edit", "expected_spans"), ("test", "test_file_spans")):
        for path, names in literal(task[field]).items():
            for name in names:
                i = by_key.get((path, name))
                if i is None:
                    if not UNRESOLVABLE.match(name):
                        unresolved.append(f"{kind}:{path}:{name}")
                elif i not in labels:
                    labels[i] = kind
    if not any(kind == "edit" for kind in labels.values()):
        return None
    order = bm25_rank(task["problem_statement"], index)
    rank = {i: r + 1 for r, i in enumerate(order)}
    rng = seeded(task["instance_id"])
    negatives: dict[int, str] = {}
    for i in order:
        if len([k for k in negatives.values() if k == "bm25"]) >= BM25_NEGATIVES:
            break
        if i not in labels:
            negatives[i] = "bm25"
    edited_files = {index[i]["path"] for i, kind in labels.items() if kind == "edit"}
    siblings = [i for i, blk in enumerate(index) if blk["path"] in edited_files and i not in labels and i not in negatives]
    for i in rng.sample(siblings, min(SIBLING_NEGATIVES, len(siblings))):
        negatives[i] = "sibling"
    others = [i for i in range(len(index)) if i not in labels and i not in negatives]
    for i in rng.sample(others, min(RANDOM_NEGATIVES, len(others))):
        negatives[i] = "random"
    chosen = sorted({**negatives, **labels}.items(), key=lambda item: (index[item[0]]["path"], index[item[0]]["start_line"]))
    blocks = []
    for n, (i, kind) in enumerate(chosen):
        blk = index[i]
        blocks.append(
            {
                "id": f"b{n}",
                **blk,
                "label": kind if kind in ("edit", "test") else "negative",
                "negative_kind": kind if kind not in ("edit", "test") else None,
                "bm25_rank": rank[i],
            }
        )
    record = {
        "instance_id": task["instance_id"],
        "repo": task["repo"],
        "base_commit": task["base_commit"],
        "problem_statement": task["problem_statement"],
        "indexed_blocks": len(index),
        "unresolved_spans": unresolved,
        "blocks": blocks,
    }
    return record, {blk["path"]: files[blk["path"]] for blk in blocks}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=ROOT / "evals/runs/retrieval-pairs")
    parser.add_argument("--limit", type=int, default=44)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    tasks = select_tasks(load_dataset(args.out / "cache"), args.limit)
    pairs, sources, skipped = [], {}, []
    for n, task in enumerate(tasks, 1):
        print(f"[{n}/{len(tasks)}] {task['instance_id']}", file=sys.stderr, flush=True)
        built = build_task(task, args.out / "cache")
        if built is None:
            skipped.append(task["instance_id"])
            continue
        record, files = built
        pairs.append(record)
        sources[record["instance_id"]] = files
    with open(args.out / "pairs.jsonl", "w") as handle:
        for record in pairs:
            handle.write(json.dumps(record) + "\n")
    with open(args.out / "files.jsonl", "w") as handle:
        for instance_id, files in sources.items():
            handle.write(json.dumps({"instance_id": instance_id, "files": files}) + "\n")
    summarize(pairs, skipped)


def summarize(pairs: list[dict], skipped: list[str]):
    labels = Counter(blk["label"] for rec in pairs for blk in rec["blocks"])
    kinds = Counter(blk["negative_kind"] for rec in pairs for blk in rec["blocks"] if blk["negative_kind"])
    edit_ranks = [blk["bm25_rank"] for rec in pairs for blk in rec["blocks"] if blk["label"] == "edit"]
    print(f"tasks: {len(pairs)} (skipped without a resolvable edit span: {len(skipped)})")
    print(f"blocks: {sum(labels.values())}  labels: {dict(labels)}  negatives: {dict(kinds)}")
    print(f"repos: {dict(Counter(rec['repo'] for rec in pairs))}")
    for k in (10, 50, 200):
        hit = sum(r <= k for r in edit_ranks) / max(1, len(edit_ranks))
        print(f"BM25 recall of edit blocks within top {k}: {hit:.0%}")
    tasks_hit = sum(any(b["label"] == "edit" and b["bm25_rank"] <= 50 for b in rec["blocks"]) for rec in pairs)
    print(f"tasks with at least one edit block in BM25 top 50: {tasks_hit}/{len(pairs)}")
    grouped = defaultdict(list)
    for rec in pairs:
        grouped["chars"].append(sum(len(b["source"]) for b in rec["blocks"]) + len(rec["problem_statement"]))
    print(f"characters per task (statement + blocks): median {sorted(grouped['chars'])[len(pairs) // 2]}")


if __name__ == "__main__":
    main()
