"""
sk2_kg.py  --  TASK 2, part 2

Query layer over the Sigma field-vocabulary knowledge graph.

Reads from Neo4j when it is reachable, and falls back to the mined JSON when it
is not. Either way it RECORDS which backend served the query, so a run is
reproducible and the paper can state honestly whether the graph was live.

This is what replaces `graph_engine.py`. The old one queried MITRE Tool names
with detection values and returned data the assembler never read. This one
answers the three questions that actually constrain generation:

    route_logsource(techniques, hints)  -> (product, category, service)
    allowed_fields(logsource)           -> ordered list of legitimate fields
    allowed_modifiers(logsource, field) -> modifiers that field really accepts
    suggest_modifier(logsource, field, value) -> shape-based choice
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

log = logging.getLogger(__name__)


def _shape(v) -> str:
    if isinstance(v, bool): return "bool"
    if isinstance(v, int):  return "int"
    if isinstance(v, list): return "list"
    if v is None:           return "null"
    s = str(v)
    if re.fullmatch(r"\d{1,3}(\.\d{1,3}){3}(/\d{1,2})?", s):  return "ip"
    if re.fullmatch(r"[0-9a-fA-F]{32,64}", s):                return "hash"
    if s.startswith(("http://", "https://")):                 return "url"
    if s.startswith(("\\", "/")) or ":\\" in s:               return "path"
    if "*" in s:                                              return "wildcard"
    return "token"


class SigmaKG:
    def __init__(self, json_path: str | Path = "kg_out/sigma_kg.json",
                 neo4j_cfg: dict | None = None):
        self.backend = "none"
        self.driver = None
        self.graph = None
        self.query_count = 0

        p = Path(json_path)
        if p.exists():
            self.graph = json.loads(p.read_text(encoding="utf-8"))
            self._index = {ls["key"]: ls for ls in self.graph["logsources"]}
            self._triples = self.graph["logsource_triples"]
            self._tech = self.graph.get("technique_to_logsource", {})
            self.backend = "json"
        else:
            self._index, self._triples, self._tech = {}, [], {}

        cfg = neo4j_cfg or {}
        if cfg.get("enabled"):
            try:
                from neo4j import GraphDatabase
                self.driver = GraphDatabase.driver(
                    cfg.get("uri", "neo4j://127.0.0.1:7687"),
                    auth=(cfg.get("user", "neo4j"), cfg.get("password", "")))
                with self.driver.session() as s:
                    n = s.run("MATCH (l:SigmaLogSource) RETURN count(l) AS n").single()["n"]
                if n > 0:
                    self.backend = "neo4j"
                    log.info("[KG] Neo4j live with %d logsource nodes", n)
                else:
                    log.warning("[KG] Neo4j reachable but empty - load sigma_kg.cypher first. "
                                "Falling back to JSON.")
                    self.driver = None
            except Exception as e:
                log.warning("[KG] Neo4j unavailable (%s). Using JSON.", type(e).__name__)
                self.driver = None

        if self.backend == "none":
            log.error("[KG] No knowledge graph available. Run sigma_kg_miner.py first.")

    # ---------------- state, for the run record ----------------

    def status(self) -> dict:
        return {"backend": self.backend,
                "logsources": len(self._index),
                "queries": self.query_count}

    # ---------------- routing ----------------

    def route_logsource(self, techniques=None, hints=None):
        """
        Pick a (product, category, service) triple.

        techniques : ['t1059.001', ...]  from the extractor
        hints      : free-text tokens seen in the report ('aws', 'powershell', ...)

        Deterministic: scores every observed triple by KG evidence and returns
        the best. No LLM involved, so this is reproducible and ablatable.
        """
        self.query_count += 1
        techniques = [t.lower().replace("attack.", "") for t in (techniques or [])]
        hints = [str(h).lower() for h in (hints or [])]

        scores = {}
        for tri in self._triples:
            key_parts = [tri.get("product"), tri.get("category"), tri.get("service")]
            key_parts = [k for k in key_parts if k]
            if not key_parts:
                continue
            tup = (tri.get("product"), tri.get("category"), tri.get("service"))
            # prior: how common is this triple in SigmaHQ
            score = 0.10 * (tri["count"] ** 0.5)
            # hint evidence: does the report mention the product/category/service
            for part in key_parts:
                if any(part == h or part in h or h in part for h in hints):
                    score += 6.0
            scores[tup] = scores.get(tup, 0.0) + score

        # technique evidence, via the mined technique -> logsource edges
        for t in techniques:
            for key, n in self._tech.get(t, {}).items():
                for tup in list(scores):
                    parts = [p for p in tup if p]
                    if key in ("|".join(parts), *parts):
                        scores[tup] += 3.0 * (n ** 0.5)

        if not scores:
            return (None, None, None)
        best = max(scores.items(), key=lambda kv: kv[1])[0]
        return best

    # ---------------- candidate menu + snapping ----------------

    # Log sources whose "vocabulary" is really a grab-bag. windows/security has
    # 91 distinct fields because it is the Windows Security event log, i.e. a
    # hundred unrelated event types wearing one name. Including it makes the
    # field-selection task incomparable to any other log source, so scoped runs
    # exclude it by default and say so.
    GRAB_BAG = {("windows", None, "security")}

    def scope_logsources(self, top: int = 7, exclude_grab_bag: bool = True) -> list[tuple]:
        """
        The `top` most frequent log sources, optionally minus the grab-bags.

        This is how you define a defensible benchmark scope. LLMCloudHunter
        restricted itself to a single log source (aws/cloudtrail, 23 fields);
        restricting to the head of the SigmaHQ distribution is the same move,
        made explicit and reported.
        """
        out = []
        for tup in self.candidate_logsources(top=top + 8):
            if exclude_grab_bag and tup in self.GRAB_BAG:
                continue
            out.append(tup)
            if len(out) >= top:
                break
        return out

    def logsource_coverage(self, scope: list[tuple]) -> float:
        """What fraction of all SigmaHQ rules the scope accounts for."""
        want = {tuple(x or None for x in t) for t in scope}
        tot = sum(t.get("count", 0) for t in self._triples) or 1
        hit = sum(t.get("count", 0) for t in self._triples
                  if (t.get("product"), t.get("category"), t.get("service")) in want)
        return hit / tot

    def candidate_logsources(self, top: int = 40) -> list[tuple]:
        """
        The `top` most common (product, category, service) triples in SigmaHQ,
        most frequent first.

        This is what the KG offers the model to CHOOSE from, instead of the KG
        deciding alone. Measured on 17 held-out reports, keyword-based routing
        scored 17.6% and never beat the majority-class baseline of 52.9%,
        because a Windows threat report mentions 'aws', 'registry' and 'smb' in
        passing and those tokens dominated the score. The report's meaning is
        what picks the log source, and only the model reads meaning. The graph's
        job is to bound the choice and then enforce the vocabulary.
        """
        self.query_count += 1
        ranked = sorted(self._triples, key=lambda t: -t.get("count", 0))
        out = []
        for t in ranked:
            tup = (t.get("product"), t.get("category"), t.get("service"))
            if any(tup) and tup not in out:
                out.append(tup)
            if len(out) >= top:
                break
        return out

    # Synonyms the model reaches for that are not the SigmaHQ spelling.
    # This is the symbolic layer doing what it is good at: normalising names.
    CATEGORY_ALIASES = {
        "dll_load": "image_load", "module_load": "image_load",
        "library_load": "image_load", "load_module": "image_load",
        "image_loaded": "image_load",
        "file_create": "file_event", "file_write": "file_event",
        "file_creation": "file_event", "file_drop": "file_event",
        "filecreate": "file_event",
        "process_create": "process_creation", "process_start": "process_creation",
        "process_launch": "process_creation", "processcreate": "process_creation",
        "execution": "process_creation", "command_execution": "process_creation",
        "registry": "registry_set", "registry_event": "registry_set",
        "registry_write": "registry_set", "registry_modification": "registry_set",
        "powershell": "ps_script", "powershell_script": "ps_script",
        "script_block": "ps_script", "scriptblock": "ps_script",
        "network": "network_connection", "network_conn": "network_connection",
        "connection": "network_connection", "dns": "dns_query",
        "pipe": "pipe_created", "named_pipe": "pipe_created",
        "wmi": "wmi_event", "driver": "driver_load",
    }

    def _fuzzy_category(self, category):
        """`dll_load` -> `image_load` by shared tokens, when no alias applies."""
        if not category:
            return None
        want = set(str(category).lower().replace("-", "_").split("_"))
        best, best_score = None, 0.0
        seen = {t.get("category") for t in self._triples if t.get("category")}
        for real in seen:
            have = set(str(real).lower().split("_"))
            inter = want & have
            if not inter:
                continue
            score = len(inter) / max(1, len(want | have))
            if score > best_score:
                best, best_score = real, score
        return best if best_score >= 0.30 else None

    def _best_by_field_coverage(self, candidates, proposed_fields):
        """
        Among candidate triples, prefer the one whose real vocabulary explains
        the most of the fields the model actually wanted to use.

        This is the graph reasoning about evidence rather than popularity: if
        the model asked for ImageLoaded, the log source that admits ImageLoaded
        is the better answer, whatever the category was called.
        """
        if not proposed_fields:
            return None
        want = {str(f).split("|")[0].strip().lower() for f in proposed_fields if str(f).strip()}
        if not want:
            return None
        best, best_cov, best_n = None, 0.0, -1
        for tp, tc, ts, n in candidates:
            allowed = {a.lower() for a in self.allowed_fields(tp, tc, ts, top=60)}
            if not allowed:
                continue
            cov = len(want & allowed) / len(want)
            if cov > best_cov or (cov == best_cov and cov > 0 and n > best_n):
                best, best_cov, best_n = (tp, tc, ts), cov, n
        return best if best_cov > 0 else None

    def snap_logsource(self, product=None, category=None, service=None,
                       proposed_fields=None):
        """
        Take whatever the model proposed and pull it onto the nearest triple
        that really exists in SigmaHQ.

        Returns (product, category, service, snapped) where `snapped` is True
        if we had to change the model's answer. That flag is reported, not
        hidden - it is a measurement of how often the model invents a log
        source, which is a number no CTI-to-Sigma paper currently gives.
        """
        self.query_count += 1
        p = (product or None) and str(product).strip().lower() or None
        c = (category or None) and str(category).strip().lower() or None
        s = (service or None) and str(service).strip().lower() or None

        known = [(t.get("product"), t.get("category"), t.get("service"), t.get("count", 0))
                 for t in self._triples]

        def norm(x):
            return (x or None) and str(x).strip().lower() or None

        # 1. exact triple
        for tp, tc, ts, _ in known:
            if (norm(tp), norm(tc), norm(ts)) == (p, c, s):
                return (tp, tc, ts, False)

        # 1b. normalise a category the model spelled its own way, then retry
        if c:
            alias = self.CATEGORY_ALIASES.get(c) or self._fuzzy_category(c)
            if alias and alias != c:
                for tp, tc, ts, _ in known:
                    if (norm(tp), norm(tc), norm(ts)) == (p, alias, s):
                        return (tp, tc, ts, True)
                c = alias

        # 2. same product + category (ignore an invented service)
        best = None
        for tp, tc, ts, n in known:
            if p and c and norm(tp) == p and norm(tc) == c:
                if best is None or n > best[3]:
                    best = (tp, tc, ts, n)
        if best:
            return (best[0], best[1], best[2], True)

        # 3. same product + service
        best = None
        for tp, tc, ts, n in known:
            if p and s and norm(tp) == p and norm(ts) == s:
                if best is None or n > best[3]:
                    best = (tp, tc, ts, n)
        if best:
            return (best[0], best[1], best[2], True)

        # 4. same category anywhere (categories are the portable part of Sigma)
        best = None
        for tp, tc, ts, n in known:
            if c and norm(tc) == c:
                if best is None or n > best[3]:
                    best = (tp, tc, ts, n)
        if best:
            return (best[0], best[1], best[2], True)

        # 5. same product - but let the requested FIELDS choose which one,
        #    falling back to the most common if the fields say nothing.
        same_product = [(tp, tc, ts, n) for tp, tc, ts, n in known
                        if p and norm(tp) == p]
        if same_product:
            by_fields = self._best_by_field_coverage(same_product, proposed_fields)
            if by_fields:
                return (by_fields[0], by_fields[1], by_fields[2], True)
            best = max(same_product, key=lambda k: k[3])
            return (best[0], best[1], best[2], True)

        # 6. give up: the single most common log source in SigmaHQ
        if known:
            tp, tc, ts, _ = max(known, key=lambda k: k[3])
            return (tp, tc, ts, True)
        return (product, category, service, False)

    # ---------------- vocabulary ----------------

    def _lookup_keys(self, product, category, service):
        p = (product or "").lower() or None
        c = (category or "").lower() or None
        s = (service or "").lower() or None
        keys = []
        if p and c and s: keys.append(f"{p}|{c}|{s}")
        if p and c:       keys.append(f"{p}|{c}")
        if p and s:       keys.append(f"{p}|{s}")
        if c and s:       keys.append(f"{c}|{s}")
        if c:             keys.append(c)
        if s:             keys.append(s)
        if p:             keys.append(p)
        return keys

    def allowed_fields(self, product=None, category=None, service=None, top=None):
        """Ordered by how often the field really occurs under that logsource."""
        self.query_count += 1
        if self.driver is not None:
            for key in self._lookup_keys(product, category, service):
                with self.driver.session() as sess:
                    rows = sess.run(
                        "MATCH (l:SigmaLogSource {key:$k})-[r:HAS_FIELD]->(f:SigmaField) "
                        "RETURN f.name AS name, r.count AS c ORDER BY r.count DESC", k=key)
                    names = [r["name"] for r in rows]
                if names:
                    return names[:top] if top else names
            return []

        for key in self._lookup_keys(product, category, service):
            ls = self._index.get(key)
            if ls and ls["fields"]:
                names = [f["name"] for f in ls["fields"]]
                return names[:top] if top else names
        return []

    def allowed_modifiers(self, field, product=None, category=None, service=None):
        self.query_count += 1
        for key in self._lookup_keys(product, category, service):
            ls = self._index.get(key)
            if not ls:
                continue
            for f in ls["fields"]:
                if f["name"].lower() == str(field).lower():
                    return [m for m in f["modifiers"] if m != "<none>"]
        return []

    def suggest_modifier(self, field, value, product=None, category=None, service=None):
        """
        Choose a modifier from the ones this field actually accepts, using the
        shape of the value. Falls back to a bare match.
        """
        allowed = set(self.allowed_modifiers(field, product, category, service))
        sh = _shape(value if not isinstance(value, list) else (value[0] if value else ""))
        if sh == "path" and "endswith" in allowed:      return "endswith"
        if sh == "url" and "contains" in allowed:       return "contains"
        if sh == "wildcard":                            return ""
        if sh == "token" and "contains" in allowed and "endswith" not in allowed:
            return "contains"
        return ""

    def field_is_valid(self, field, product=None, category=None, service=None) -> bool:
        base = str(field).split("|")[0].strip().lower()
        return base in {f.lower() for f in self.allowed_fields(product, category, service)}

    def close(self):
        if self.driver is not None:
            try: self.driver.close()
            except Exception: pass
