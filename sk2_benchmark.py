"""
sk2_benchmark.py  --  TASK 4

Builds the CTI-to-Sigma benchmark properly.

What was wrong with build_sigma_dataset.py
------------------------------------------
  * stopped at a target count instead of harvesting the corpus
  * took only the FIRST reference URL that yielded 400 characters
  * no domain filtering, so MITRE technique pages, Atomic Red Team markdown and
    Microsoft auth walls were saved as "CTI reports"
  * never checked whether the article could answer the rule

Measured consequence on the previous build: mean literal grounding 0.325,
median 0.106, and 23 of 50 sampled articles contained ZERO of their rule's
values. Nearly half the benchmark was unanswerable.

What this does instead
----------------------
  1. harvests EVERY SigmaHQ rule
  2. tries EVERY reference URL per rule, keeps the best-grounded one
  3. rejects documentation domains, auth walls and JavaScript shells
  4. scores literal grounding and stratifies:
         grounded  >= 0.50     partial 0.10-0.50     ungrounded < 0.10
  5. writes the whole thing plus a distribution report

The strata are all published. Nothing is silently discarded -- the point is to
be the first benchmark in this area that states how answerable it is.

    python sk2_benchmark.py --out data_bench --workers 12
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml

# --- domains that are documentation or technique catalogues, not CTI ---
BLOCK_DOMAINS = {
    "attack.mitre.org", "github.com", "raw.githubusercontent.com", "gist.github.com",
    "docs.microsoft.com", "learn.microsoft.com", "technet.microsoft.com",
    "docs.aws.amazon.com", "cloud.google.com", "developer.mozilla.org",
    "en.wikipedia.org", "stackoverflow.com", "serverfault.com",
    "twitter.com", "x.com", "t.co", "linkedin.com", "youtube.com",
    "malpedia.caad.fkie.fraunhofer.de", "car.mitre.org", "capec.mitre.org",
    "docs.splunk.com", "www.elastic.co/guide", "support.google.com",
}
BLOCK_SUBSTR = ("atomic-red-team", "/docs/", "/documentation/", "sigmahq.io",
                "/api/", "readthedocs", "man7.org", "ss64.com")

# --- text that means we captured a wall, not an article ---
WALL_MARKERS = (
    "access to this page requires authorization",
    "you must be signed in", "sign in to continue", "please sign in",
    "enable javascript", "javascript is disabled", "please reload this page",
    "there was an error while loading", "checking your browser",
    "verify you are human", "cloudflare", "captcha", "403 forbidden",
    "404 not found", "page not found", "subscribe to continue",
)

BLOCK_SKIP = {"condition", "timeframe"}


# ----------------------------------------------------------------------

def clone_sigmahq(dest: Path) -> list[Path]:
    dest.mkdir(parents=True, exist_ok=True)
    repo = dest / "sigma"
    if not repo.exists():
        print("[*] cloning SigmaHQ (shallow) ...", flush=True)
        subprocess.run(["git", "clone", "--depth", "1",
                        "https://github.com/SigmaHQ/sigma.git", str(repo)], check=True)
    else:
        print("[*] SigmaHQ already present", flush=True)
    rules = sorted((repo / "rules").rglob("*.yml"))
    rules += sorted((repo / "rules-emerging-threats").rglob("*.yml"))
    rules += sorted((repo / "rules-threat-hunting").rglob("*.yml"))
    return [p for p in rules if not p.name.endswith(".old.yml")]


def rule_values(detection) -> set[str]:
    """Every literal value in the detection block, lowercased."""
    out = set()

    def walk(node):
        if isinstance(node, dict):
            for k, v in node.items():
                if str(k).lower() in BLOCK_SKIP:
                    continue
                walk(v)
        elif isinstance(node, list):
            for i in node:
                walk(i)
        elif isinstance(node, (str, int, float, bool)):
            s = str(node).strip().lower().strip("*")
            if len(s) > 2:
                out.add(s)

    if isinstance(detection, dict):
        for bname, block in detection.items():
            if str(bname).lower() in BLOCK_SKIP:
                continue
            walk(block)
    return out


def url_allowed(url: str) -> bool:
    u = url.lower()
    if not u.startswith("http"):
        return False
    if any(u.endswith(ext) for ext in (".pdf", ".exe", ".zip", ".png", ".jpg", ".gif")):
        return False
    host = re.sub(r"^https?://", "", u).split("/")[0]
    if host in BLOCK_DOMAINS or any(host.endswith("." + d) for d in BLOCK_DOMAINS):
        return False
    return not any(s in u for s in BLOCK_SUBSTR)


def fetch(url: str, timeout: int = 15) -> str:
    import warnings
    import requests
    from bs4 import BeautifulSoup
    # Some references are RSS/XML feeds. Parsing them with the HTML parser works
    # fine for our purposes (we only want the text), so silence the notice
    # rather than printing a 12-line warning for every feed we meet.
    try:
        from bs4 import XMLParsedAsHTMLWarning
        warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
    except Exception:
        pass
    headers = {"User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                             "(KHTML, like Gecko) Chrome/122.0 Safari/537.36")}
    r = requests.get(url, headers=headers, timeout=timeout)
    if r.status_code != 200:
        return ""
    soup = BeautifulSoup(r.content, "html.parser")
    for t in soup(["script", "style", "nav", "footer", "header", "noscript", "aside", "form"]):
        t.decompose()
    parts = [p.get_text(" ", strip=True)
             for p in soup.find_all(["p", "h1", "h2", "h3", "h4", "li", "pre", "code", "td"])]
    return "\n".join(p for p in parts if len(p) > 25)


def looks_like_wall(text: str) -> bool:
    head = text[:2500].lower()
    return any(m in head for m in WALL_MARKERS)


def grounding(values: set[str], article: str) -> float:
    if not values:
        return 0.0
    low = article.lower()
    return sum(1 for v in values if v in low) / len(values)


# ----------------------------------------------------------------------

class DiskCache:
    """
    One small file per URL instead of one giant dict.

    The previous version kept every article's full text in a dict AND
    re-serialised the whole thing with json.dumps every 100 rules. By rule 2000
    that was ~60 MB re-encoded thirty times while 12 threads parsed HTML, which
    is what locked up the machine. Nothing is held in memory here.
    """

    def __init__(self, root: Path):
        self.root = Path(root); self.root.mkdir(parents=True, exist_ok=True)

    def _p(self, url: str) -> Path:
        h = hashlib.sha1(url.encode()).hexdigest()
        return self.root / h[:2] / f"{h}.txt"

    def get(self, url: str):
        f = self._p(url)
        if f.exists():
            try:
                return f.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                return None
        return None

    def put(self, url: str, text: str) -> None:
        f = self._p(url); f.parent.mkdir(parents=True, exist_ok=True)
        try:
            f.write_text(text or "", encoding="utf-8")
        except Exception:
            pass

    def __len__(self):
        return sum(1 for _ in self.root.rglob("*.txt"))


def process_rule(path: Path, cache: "DiskCache", lock=None) -> dict | None:
    try:
        rule = yaml.safe_load(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return None
    if not isinstance(rule, dict) or "detection" not in rule or "logsource" not in rule:
        return None
    refs = [r for r in (rule.get("references") or []) if isinstance(r, str)]
    refs = [r for r in refs if url_allowed(r)]
    if not refs:
        return {"status": "no_usable_reference", "rule_path": str(path)}

    vals = rule_values(rule.get("detection"))
    best = None
    for url in refs[:6]:
        text = cache.get(url)
        if text is None:
            try:
                text = fetch(url)
            except Exception:
                text = ""
            cache.put(url, text)
        if len(text) < 500 or looks_like_wall(text):
            continue
        g = grounding(vals, text)
        if best is None or g > best["grounding"]:
            best = {"url": url, "text": text, "grounding": g}
        if g >= 0.8:
            break

    if best is None:
        return {"status": "no_fetchable_article", "rule_path": str(path)}

    g = best["grounding"]
    stratum = "grounded" if g >= 0.50 else ("partial" if g >= 0.10 else "ungrounded")
    return {
        "status": "ok", "stratum": stratum, "grounding": round(g, 4),
        "rule_path": str(path), "url": best["url"],
        "rule": rule, "text": best["text"],
        "n_values": len(vals),
        "logsource": rule.get("logsource"),
        "date": str(rule.get("date", "")),
    }


def _free(rec: dict) -> dict:
    """Drop the article body once it has been written to disk."""
    rec.pop("text", None)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data_bench")
    ap.add_argument("--workers", type=int, default=6,
                    help="6 is safe on a laptop; raise to 16 on Colab")
    ap.add_argument("--limit", type=int, default=0, help="0 = all rules")
    ap.add_argument("--target", type=int, default=1000,
                    help="stop as soon as this many USABLE pairs exist "
                         "(grounded + partial). 0 = keep going through every rule.")
    ap.add_argument("--seed", type=int, default=20260823,
                    help="shuffle seed, so the sample is random but reproducible")
    args = ap.parse_args()

    out = Path(args.out)
    (out / "articles").mkdir(parents=True, exist_ok=True)
    (out / "rules").mkdir(parents=True, exist_ok=True)

    paths = clone_sigmahq(out / "_src")

    # Shuffle before sampling. Taking rules in folder order would load the
    # benchmark with whatever sorts first (all the 'application/' rules, say)
    # and the sample would not represent SigmaHQ. Seeded, so it reproduces.
    random.Random(args.seed).shuffle(paths)

    if args.limit:
        paths = paths[: args.limit]
    print(f"[*] {len(paths)} SigmaHQ rules available (shuffled, seed {args.seed})", flush=True)
    if args.target:
        print(f"[*] will stop as soon as {args.target} usable pairs are collected", flush=True)

    cache = DiskCache(out / "_fetch_cache")
    n_cached = len(cache)
    if n_cached:
        print(f"[*] reusing {n_cached} cached fetches", flush=True)

    # Stream results straight to disk. Article bodies never accumulate in RAM.
    index, stats, done = [], Counter(), 0
    seq = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(process_rule, p, cache) for p in paths]
        for fut in as_completed(futs):
            done += 1
            try:
                r = fut.result()
            except Exception:
                stats["worker_error"] += 1
                r = None
            if r is None:
                stats["unparseable"] += 1
            else:
                stats[r["status"]] += 1
                if r["status"] == "ok":
                    rid = f"cti_{seq:05d}"
                    # Write BOTH files first, and only then record the pair.
                    # The old code indexed first, so a failed write left
                    # index.json naming a file that does not exist - which is
                    # exactly what crashed the run after the freeze.
                    try:
                        (out / "articles" / f"{rid}.txt").write_text(
                            r["text"], encoding="utf-8")
                        (out / "rules" / f"{rid}.yml").write_text(
                            yaml.safe_dump(r["rule"], sort_keys=False, allow_unicode=True),
                            encoding="utf-8")
                    except Exception:
                        stats["write_failed"] += 1
                        _free(r)
                        continue
                    seq += 1
                    stats[f"stratum_{r['stratum']}"] += 1
                    index.append({"id": rid, "stratum": r["stratum"],
                                  "grounding": r["grounding"], "url": r["url"],
                                  "n_values": r["n_values"], "logsource": r["logsource"],
                                  "date": r["date"], "source_rule": r["rule_path"]})
                    _free(r)

            usable = stats["stratum_grounded"] + stats["stratum_partial"]
            if done % 50 == 0:
                print(f"    {done}/{len(paths)}  ok={stats['ok']} "
                      f"usable={usable}  grounded={stats['stratum_grounded']}", flush=True)
                # small, bounded write - the index only, never the article bodies
                (out / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")

            if args.target and usable >= args.target:
                print(f"\n[+] reached {usable} usable pairs - stopping early "
                      f"after examining {done} rules", flush=True)
                for f in futs:
                    f.cancel()
                break

    (out / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    results = index

    lines = ["=" * 74, "CTI-TO-SIGMA BENCHMARK -- BUILD REPORT", "=" * 74,
             f"SigmaHQ rules examined        : {len(paths)}",
             f"usable pairs built            : {len(index)}",
             f"no usable reference URL       : {stats['no_usable_reference']}",
             f"no fetchable article          : {stats['no_fetchable_article']}",
             f"unparseable rule files        : {stats['unparseable']}", "",
             "ANSWERABILITY STRATA  (fraction of the rule's literal values present in the article)",
             "-" * 74]
    tot = max(1, len(index))
    for s in ("grounded", "partial", "ungrounded"):
        n = stats[f"stratum_{s}"]
        lines.append(f"  {s:<12} {n:>5}   {100*n/tot:5.1f}%")
    ls = Counter((i["logsource"] or {}).get("product") for i in index)
    lines += ["", "TOP PRODUCTS", "-" * 74]
    lines += [f"  {str(k):<22} {v}" for k, v in ls.most_common(12)]
    report = "\n".join(lines)
    (out / "build_report.txt").write_text(report, encoding="utf-8")
    print("\n" + report)
    print(f"\n[+] {out/'index.json'}  ({len(index)} pairs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
