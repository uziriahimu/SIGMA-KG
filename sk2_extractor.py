"""
sk2_extractor.py  --  TASK 1 + TASK 3

The rebuilt generation stage.

Three changes from src/extractor.py, each one deliberate:

TASK 1 -- THE TITLE IS GONE.
    The old prompt opened with "TARGET DETECTION RULE TITLE: {title}" and told
    the model to match it. Measured effect (n=50, paired Wilcoxon): the title
    contributed +0.0447 value F1 at p=0.0126, the article contributed +0.0287
    at p=0.5507. The model was reading the answer's title, not the report.
    Ground truth is now used for scoring only. Nothing about the reference rule
    reaches this file.

TASK 3 -- EXEMPLARS ARE RETRIEVED, NOT FIXED.
    Two hardcoded examples (Bitbucket, Mimikatz) are replaced by the k nearest
    TRAINING rules, so a macOS report is shown macOS conventions.

TASK 2 -- THE FIELD VOCABULARY IS ENFORCED.
    The KG routes the log source, then supplies the admissible field list. The
    model chooses from that list. Anything outside it is dropped and counted,
    never silently kept. Field validity therefore becomes an architectural
    guarantee, and we report it as one.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a senior detection engineer writing SigmaHQ rules from threat intelligence.

You read a CTI report and decide two things: WHICH log source would record this activity, and WHICH concrete observables to detect on.

Return STRICTLY VALID JSON:
{
  "logsource": {"product": "<or null>", "category": "<or null>", "service": "<or null>"},
  "logsource_reason": "<one short sentence: what activity in the report told you this>",
  "observables": [
    {"value": "<verbatim string copied from the report>",
     "kind": "process|command|file|path|registry|domain|ip|hash|url|api_call|user_agent|event_id|other",
     "evidence": "<the sentence you took it from, copied verbatim>"}
  ],
  "detection_selections": { "<FieldName>": ["<value>"] },
  "mitre_attack_tags": ["attack.execution", "attack.t1059.001"]
}

HARD RULES:
1. Every "value" MUST appear verbatim in the report. If you cannot copy it, omit it.
   This rule is absolute - it is what stops invention.
2. Within that rule, BE COMPLETE. A real Sigma rule usually lists several values.
   Extract EVERY observable the report gives that this detection would need:
   every file name, every command line, every path, every registry key, every
   domain, every hash. Missing a real observable is as much an error as
   inventing one.
3. Do not pad with empty lists or placeholders. Every entry must carry evidence.
4. Group values under the field that fits them: several file names belong in one
   field as a list, not spread across invented fields.
5. Choose the log source that would really record this activity, and field names
   appropriate to it."""


# The schema knowledge below is exactly what the knowledge graph contributes.
# It is therefore attached ONLY when use_kg is on. It used to live in the shared
# system prompt, which meant the "no_kg" baseline was handed the graph's own
# knowledge through the prompt - the ablation was comparing two DELIVERY
# MECHANISMS for the same information, not presence versus absence of it. That
# is why no_kg matched or beat full on the first full run.
KG_SCHEMA_GUIDANCE = """CHOOSING THE LOG SOURCE - this decides everything downstream, so think about it first:
  a process was started            -> category process_creation
  a file was written or dropped    -> category file_event
  a registry value was set         -> category registry_set
  a DLL or module was loaded       -> category image_load
  a PowerShell script ran          -> category ps_script
  an outbound connection was made  -> category network_connection
Pick the category describing the ACTIVITY the report documents, not merely a
word it happens to mention. A report can mention the registry while documenting
a process launch; then the answer is process_creation.

FIELD CONVENTIONS for the log source you choose:
  process_creation -> Image, CommandLine, ParentImage, OriginalFileName
  file_event       -> TargetFilename, Image
  registry_set     -> TargetObject, Details
  image_load       -> ImageLoaded, Image
  ps_script        -> ScriptBlockText
  network_connection -> DestinationIp, DestinationHostname, DestinationPort
"""


def _hints_from_text(text: str, limit: int = 400) -> list[str]:
    """Cheap surface hints for KG routing. No model call."""
    t = (text or "").lower()
    vocab = [
        "aws", "cloudtrail", "s3", "iam", "lambda", "ec2", "azure", "entra", "m365",
        "office 365", "gcp", "kubernetes", "k8s", "windows", "powershell", "sysmon",
        "linux", "auditd", "macos", "github", "gitlab", "bitbucket", "okta", "proxy",
        "dns", "firewall", "antivirus", "registry", "process creation", "webserver",
        "cloudflare", "zeek", "sysinternals", "wmi", "rdp", "smb", "ldap",
    ]
    hits = [v for v in vocab if v in t]
    caps = re.findall(r"\b[A-Z][a-zA-Z]{3,}[A-Z][a-zA-Z]{2,}\b", text or "")  # CamelCase API names
    return (hits + [c.lower() for c in caps[:20]])[:limit]


class SchemaGroundedExtractor:
    """
    KG-grounded, retrieval-augmented, title-free extraction.

    `kg`    : sk2_kg.SigmaKG
    `index` : sk2_retrieval.ExemplarIndex or None
    """

    def __init__(self, kg, index=None, model_name: str = "gpt-4o-mini",
                 api_key: str | None = None, k_exemplars: int = 5,
                 max_chars: int = 12000, use_kg: bool = True, use_retrieval: bool = True,
                 scope: list | None = None, base_url: str | None = None,
                 menu_top: int = 120,
                 schema_in_prompt: bool = False, enforce_fields: bool = True):
        import openai
        # base_url lets the SAME pipeline run against an open-source model served
        # through any OpenAI-compatible endpoint (vLLM, Ollama, Together,
        # OpenRouter, DeepInfra). Two reviewers asked for open-source results;
        # this is the switch that produces them without touching anything else.
        base_url = base_url or os.environ.get("OPENAI_BASE_URL") or None
        self.client = openai.OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            base_url=base_url, timeout=180.0)
        self.base_url = base_url
        self.model = model_name
        self.kg = kg
        self.index = index
        self.k = k_exemplars
        self.max_chars = max_chars
        self.use_kg = use_kg
        self.use_retrieval = use_retrieval
        # When a scope is given, the menu IS the scope: the benchmark is
        # declared to cover these log sources and nothing else, the same
        # way LLMCloudHunter declares itself aws/cloudtrail only.
        self.scope = list(scope) if scope else None
        # 40 was too few: on the unscoped run, 14 of the 41 reference log
        # sources were absent from the menu, so 6% of the test set was
        # unreachable by construction while the unconstrained baseline could
        # still name them. The menu must cover the label space.
        self.menu_top = menu_top
        # ------------------------------------------------------------------
        # The reviewer question this exists to answer: could you get the same
        # reduction by simply putting the legal field names in the prompt,
        # with no symbolic enforcement at all?
        #
        #   schema_in_prompt  list every allowed field for every in-scope log
        #                     source in the prompt. For the seven-source scope
        #                     that is 64 names, about 200 tokens, so context
        #                     length is NOT what makes prompting insufficient.
        #                     Do not argue that it is.
        #   enforce_fields    run the post-hoc deletion. Turning this off with
        #                     schema_in_prompt on gives the purely generative
        #                     condition: the model is TOLD the schema and
        #                     nothing checks that it obeyed.
        #
        # Defaults reproduce the published behaviour exactly.
        # ------------------------------------------------------------------
        self.schema_in_prompt = schema_in_prompt
        self.enforce_fields = enforce_fields

    # ---------------------------------------------------------------

    def _schema_block(self) -> str:
        """The allowed field list for every log source on the menu."""
        menu = self.scope or self.kg.candidate_logsources(top=self.menu_top)
        lines = []
        for p, c, s in menu:
            fields = self.kg.allowed_fields(p, c, s, top=40)
            if not fields:
                continue
            key = "/".join(str(x) for x in (p, c, s) if x)
            lines.append(f"  {key}: " + ", ".join(fields))
        if not lines:
            return ""
        return ("THESE ARE THE ONLY FIELD NAMES THAT EXIST FOR EACH LOG SOURCE. "
                "Use no other field name under any circumstances:\n"
                + "\n".join(lines) + "\n\n")

    def generate(self, report_text: str, report_id: str | None = None) -> dict:
        """
        Returns a dict with the parsed IR plus a full provenance record.
        NOTHING derived from the reference rule enters this method.
        """
        rec: dict[str, Any] = {
            "report_id": report_id, "used_kg": self.use_kg,
            "used_retrieval": self.use_retrieval and self.index is not None,
            "api_error": None, "dropped_fields": [], "ungrounded_values": [],
        }

        # --- the KG offers a MENU; the model chooses from it --------------
        # It used to be the other way round: the KG picked the log source from
        # keyword hints before the model saw anything, and told the model to
        # use it. Measured on 17 held-out reports that scored 17.6%, against a
        # 52.9% majority-class baseline - so the routing was actively harmful,
        # and every wrong route poisoned the field list and the values with it.
        if self.use_kg:
            menu = self.scope or self.kg.candidate_logsources(top=self.menu_top)
            menu_txt = "\n".join(
                f"  product={p or 'null'}  category={c or 'null'}  service={s or 'null'}"
                for p, c, s in menu)
            menu_block = (
                KG_SCHEMA_GUIDANCE + "\n"
                "CHOOSE ONE OF THESE LOG SOURCES (they are the ones that really "
                "exist in SigmaHQ, most common first):\n" + menu_txt + "\n\n")
            if self.schema_in_prompt:
                menu_block += self._schema_block()
        else:
            menu_block = ""

        # --- retrieved exemplars ---------------------------------------
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        if self.use_retrieval and self.index is not None:
            try:
                from sk2_retrieval import exemplar_to_prompt
                neighbours = self.index.retrieve(report_id, report_text, k=self.k)
                rec["exemplars"] = [{"id": n["id"], "score": round(n["score"], 4)} for n in neighbours]
                for n in neighbours:
                    u, a = exemplar_to_prompt(n)
                    messages.append({"role": "user", "content": u})
                    messages.append({"role": "assistant", "content": a})
            except Exception as e:
                # A LEAKAGE GUARD failure means this report is sitting in its
                # own exemplar pool. That is a correctness bug, not a hiccup:
                # swallowing it would silently downgrade `full` into
                # `no_retrieval` for that report and you would pay for a run
                # whose configuration is not what the paper claims. Crash loud.
                if "LEAKAGE GUARD" in str(e):
                    raise
                log.warning("retrieval failed for %s: %s", report_id, e)
                rec["exemplars"] = []
                rec["used_retrieval"] = False
        else:
            rec["exemplars"] = []

        user = (
            f"{menu_block}"
            f"CTI REPORT:\n{(report_text or '')[:self.max_chars]}"
        )
        messages.append({"role": "user", "content": user})
        rec["prompt_chars"] = sum(len(m["content"]) for m in messages)

        # --- call -------------------------------------------------------
        data = None
        for attempt in range(3):
            try:
                kw = dict(model=self.model, messages=messages, temperature=0.0)
                # Not every OpenAI-compatible server supports JSON mode. Ask for it
                # first; on the later attempts fall back and parse the text.
                if attempt == 0:
                    kw["response_format"] = {"type": "json_object"}
                r = self.client.chat.completions.create(**kw)
                raw = r.choices[0].message.content or ""
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    m = re.search(r"\{.*\}", raw, re.S)   # strip ``` fences / preamble
                    if not m:
                        raise
                    data = json.loads(m.group(0))
                break
            except Exception as e:
                rec["api_error"] = f"{type(e).__name__}: {e}"
                time.sleep(2 * (attempt + 1))

        if data is None:
            rec["generation_failed"] = True
            rec["selections"] = {}
            rec["tags"] = []
            rec["observables"] = []
            return rec

        rec["generation_failed"] = False

        # --- verbatim grounding check (anti-hallucination, measurable) ---
        # The model is asked for objects like {"value": "...", "kind": "..."},
        # but it does not always comply - sometimes it returns a bare list of
        # strings, or nests them one level deeper. Accept every shape rather
        # than crashing a paid run over JSON formatting.
        def _value_of(o) -> str:
            if isinstance(o, dict):
                for key in ("value", "observable", "indicator", "text", "name"):
                    if o.get(key):
                        return str(o[key]).strip()
                return ""
            if isinstance(o, (str, int, float)):
                return str(o).strip()
            return ""

        raw_obs = data.get("observables")
        if isinstance(raw_obs, dict):          # {"observables": {...}} -> take the values
            raw_obs = list(raw_obs.values())
        if not isinstance(raw_obs, list):
            raw_obs = []

        low = (report_text or "").lower()
        obs_ok, obs_bad = [], []
        for o in raw_obs:
            if isinstance(o, list):            # one level of unwanted nesting
                for inner in o:
                    v = _value_of(inner)
                    if v and v.lower() in low:
                        obs_ok.append(inner)
                    elif v:
                        obs_bad.append(v)
                continue
            v = _value_of(o)
            if v and v.lower() in low:
                obs_ok.append(o)
            elif v:
                obs_bad.append(v)
        rec["observables"] = obs_ok
        rec["ungrounded_values"] = obs_bad
        rec["n_observables"] = len(obs_ok)

        # --- the model's log source, resolved against the graph -----------
        prop = data.get("logsource")
        if not isinstance(prop, dict):
            prop = {}
        p_prop = prop.get("product") or None
        c_prop = prop.get("category") or None
        s_prop = prop.get("service") or None
        for bad in ("null", "none", "n/a", "-", ""):
            p_prop = None if str(p_prop).strip().lower() == bad else p_prop
            c_prop = None if str(c_prop).strip().lower() == bad else c_prop
            s_prop = None if str(s_prop).strip().lower() == bad else s_prop

        rec["proposed_logsource"] = {"product": p_prop, "category": c_prop, "service": s_prop}
        rec["logsource_reason"] = str(data.get("logsource_reason") or "")[:300]

        raw_sel_peek = data.get("detection_selections")
        peek_fields = list(raw_sel_peek.keys()) if isinstance(raw_sel_peek, dict) else []

        if self.use_kg:
            product, category, service, snapped = self.kg.snap_logsource(
                p_prop, c_prop, s_prop, proposed_fields=peek_fields)
            rec["logsource_snapped"] = bool(snapped)
            # enforce_fields=False leaves `allowed` empty, which the filter
            # below reads as "accept whatever the model wrote". Routing still
            # happens; only the vocabulary check is switched off.
            allowed = (self.kg.allowed_fields(product, category, service, top=40)
                       if self.enforce_fields else [])
        else:
            product, category, service = p_prop, c_prop, s_prop
            rec["logsource_snapped"] = False
            allowed = []

        rec["routed_logsource"] = {"product": product, "category": category, "service": service}
        rec["n_allowed_fields"] = len(allowed)

        # --- field vocabulary enforcement --------------------------------
        def _nonempty(v):
            """Drop the empty padding the model emits to fill out a schema."""
            if v is None:
                return False
            if isinstance(v, (list, tuple, set, dict)):
                return len([x for x in v if str(x).strip()]) > 0
            return bool(str(v).strip())

        raw_sel = data.get("detection_selections") or {}
        clean: dict[str, Any] = {}
        allowed_lower = {a.lower(): a for a in allowed}
        for k, v in (raw_sel.items() if isinstance(raw_sel, dict) else []):
            if not _nonempty(v):
                rec["empty_fields"] = rec.get("empty_fields", []) + [str(k)]
                continue
            base = str(k).split("|")[0].strip()
            if self.use_kg and allowed_lower and base.lower() not in allowed_lower:
                rec["dropped_fields"].append(str(k))
                continue
            canonical = allowed_lower.get(base.lower(), base)
            if "|" in str(k):
                mod = str(k).split("|", 1)[1]
            else:
                mod = self.kg.suggest_modifier(canonical, v, product, category, service) if self.use_kg else ""
            clean[f"{canonical}|{mod}" if mod else canonical] = v

        rec["selections"] = clean
        rec["n_empty_fields"] = len(rec.get("empty_fields", []))
        rec["n_dropped_fields"] = len(rec["dropped_fields"])

        tags = set()
        for t in (data.get("mitre_attack_tags") or []):
            c = str(t).lower().strip().replace("_", ".")
            if not c.startswith("attack."):
                c = "attack." + c
            tags.add(c)
        rec["tags"] = sorted(tags)
        return rec
