"""G2-lite task data (generator v2): a fixed number of records embedded in digit-free filler prose, built from the frozen config.

A document is a header, then n record lines "编号 XXXX：YYYY" (one per line, 12 Qwen3 tokens each because the tokenizer
splits digits one per token) separated by paragraphs of filler sentences drawn from the config pool. Ids are distinct
four-digit numbers, values random four-digit numbers, both bound afresh per document from a seed derived from (data
seed, task, length, document index). The question is appended after the document in the same user turn; the chat
template (thinking disabled) wraps everything.

Why v2: the v1 dense table (169 / 681 records, no filler) was beyond Qwen3-0.6B on the calibration set (seed 777) under
every wording tried (A ≤ 3 / 12 at 2K); the task (return the value of a named record) is unchanged, the distractor
density is not. The calibration evidence is recorded in reports/cards/G2lite_mac_m4max.md.

The common prefix is the rendered text before the question marker. It is tokenised on its own and checked to be an
exact token prefix of the full prompt, so every read path starts from identical prefix tokens and the question never
enters the shared prefix (docs/PLAN_CURRENT.md §8, docs/tasks/05_G2_LITE.md).

Layout: the last two records are tail records placed at controlled token distances from the end (back_near 30–70,
back_far 150–250 tokens), the other n − 2 are spread evenly over the body; filler is adjusted until the prefix is within
the length tolerance. Target buckets: "front" / "middle" by fraction of the body records, "back_near" / "back_far" are
the two tail records, so the recent-window read B covers back_near at every rho and back_far only when k is large.
T2 asks for two far-apart records in a stated order (ascending position for even document indices, descending for odd).

Rewrites (counterfactuals) rebuild the whole document: "value" changes only the queried record's value; "irrelevant"
changes only one non-target record at least `irrelevant_min_distance_records` records away; "move" swaps the queried
record into the back_near tail slot. Nothing here loads the model.
"""
from __future__ import annotations

import hashlib
import json
import random

QUESTION_MARKER = "问题："
VARIANTS = ("base", "value", "irrelevant", "move")


def _rng(*parts) -> random.Random:
    return random.Random("/".join(str(p) for p in parts))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def doc_identifier(task: str, length_tag: str, doc_index: int, variant: str = "base") -> str:
    base = f"{task}-{length_tag}-{doc_index:02d}"
    return base if variant == "base" else f"{base}-{variant}"


def plan_docs(cfg: dict, tasks=None, lengths=None, docs_per_cell=None) -> list[dict]:
    """The (task, length, index, bucket plan) grid of base documents, in a fixed order."""
    data = cfg["data"]
    tasks = list(tasks or data["tasks"])
    lengths = list(lengths or data["lengths"].keys())
    per_cell = int(docs_per_cell or data["docs_per_cell"])
    out = []
    for task in tasks:
        plan = data["bucket_plan"][task]
        if len(plan) < per_cell:
            raise ValueError(f"bucket plan for {task} has {len(plan)} entries < docs_per_cell {per_cell}")
        for length_tag in lengths:
            for i in range(per_cell):
                out.append({"task": task, "length_tag": length_tag, "doc_index": i, "buckets": plan[i]})
    return out


def render_prompt(tok, content_prefix: str, question: str, *, enable_thinking: bool = False, marker: str = QUESTION_MARKER) -> tuple[str, str]:
    """Full chat-template text (generation prompt appended, thinking disabled) and the text before the question."""
    msgs = [{"role": "user", "content": content_prefix + question}]
    full = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking)
    i = full.rfind(question)
    if i < 0 or not question.startswith(marker) or marker in content_prefix:
        raise ValueError("the question must start with the marker, appear once at the end of the user turn, and the marker must not occur in the prefix")
    return full, full[:i]


def encode(tok, text: str) -> list[int]:
    return list(tok(text, add_special_tokens=False).input_ids)


def _token_spans(tok, prefix_text: str, char_spans: list[tuple[int, int]]) -> list[list[int]]:
    """[start, end) token spans in the prefix for the given character spans (fast-tokenizer offsets, linear sweep)."""
    enc = tok(prefix_text, add_special_tokens=False, return_offsets_mapping=True)
    offs = enc.offset_mapping
    out, t = [], 0
    for cs, ce in char_spans:
        while t < len(offs) and offs[t][1] <= cs:
            t += 1
        start = t
        e = t
        while e < len(offs) and offs[e][0] < ce:
            e += 1
        out.append([start, e])
    return out


class FillerStream:
    """A fixed, seeded sequence of filler sentences for one gap; `block(budget)` takes sentences from the front of the
    stream until the budget is met and then looks ahead for one sentence that fits the remainder, so the fill is tight
    (a few tokens) and monotone in the budget — the layout loop can then converge instead of re-rolling the text."""

    def __init__(self, rng: random.Random, pool: list[str], tok, cache: dict, joiner: str, length: int = 120):
        self.sents = []
        for _ in range(length):                               # no sentence twice in a row (the pool is small)
            sent = pool[rng.randrange(len(pool))]
            while self.sents and sent == self.sents[-1] and len(pool) > 1:
                sent = pool[rng.randrange(len(pool))]
            self.sents.append(sent)
        self.joiner = joiner
        for sent in self.sents:
            if sent not in cache:
                cache[sent] = len(encode(tok, sent))
        self.counts = [cache[s_] for s_ in self.sents]

    def block(self, budget: int) -> str:
        budget = max(1, int(budget))
        out, used, i = [], 0, 0
        while i < len(self.sents) and (not out or used + self.counts[i] <= budget):
            out.append(self.sents[i])
            used += self.counts[i]
            i += 1
        rem = budget - used
        if rem > 3:                                           # one more sentence that fits the remainder, if any
            for j in range(i, len(self.sents)):
                if self.counts[j] <= rem + 3:
                    out.append(self.sents[j])
                    break
        return self.joiner.join(out) + "\n\n"


def _measure(tok, cfg: dict, segments: list, question: str) -> tuple[list[int], list[int], str, str, list]:
    """Render segments (list of ("filler", text) / ("record", line)) and return prefix ids, full ids, texts, record char spans."""
    data = cfg["data"]
    content = data["header"] + "".join(t for _, t in segments) + data["table_end"]
    full_text, prefix_text = render_prompt(tok, content, question, enable_thinking=bool(cfg["prompt"]["enable_thinking"]), marker=data.get("question_marker", QUESTION_MARKER))
    head = full_text.index(data["header"]) + len(data["header"])
    spans, pos = [], head
    for kind, t in segments:
        if kind == "record":
            spans.append((pos, pos + len(t.rstrip("\n"))))
        pos += len(t)
    return encode(tok, prefix_text), encode(tok, full_text), full_text, prefix_text, spans


def layout_document(tok, cfg: dict, rng: random.Random, ids: list[str], values: list[str], question: str) -> tuple[list, dict]:
    """Place n records in filler so that the prefix has about `target_tokens` tokens and the last two records end at the
    configured distances from the end (back_far, then back_near); the other records are spread evenly over the body."""
    data, lay = cfg["data"], cfg["data"]["layout"]
    n = len(ids)
    fmt, pool, joiner = data["record_format"], data["filler_pool"], data.get("filler_joiner", "")
    lines = [fmt.format(id=i, value=v) for i, v in zip(ids, values)]
    cache = {}
    d_near = rng.randint(*lay["back_near_tokens_from_end"])
    d_far = rng.randint(*lay["back_far_tokens_from_end"])
    streams = [FillerStream(rng, pool, tok, cache, joiner) for _ in range(n + 1)]  # one fixed stream per gap (before each record, and after the last)
    per = len(encode(tok, lines[0] + lines[1])) - len(encode(tok, lines[0]))
    target = int(data["lengths"][cfg["_length_tag"]])
    base = len(_measure(tok, cfg, [("record", l) for l in lines], question)[0]) - per * n   # non-record overhead
    n_body = n - 1                                                               # gaps before the body records and before the far record
    tails = [max(1, d_far - d_near - per - 2), max(1, d_near - 2)]              # between the far and the near record; after the near record
    body_budget = target - base - per * n - sum(tails) - 2 * n
    gaps = [max(1, body_budget // n_body)] * n_body
    history = []
    for attempt in range(lay["max_adjust_iterations"]):
        segments = []
        for i in range(n_body):                                   # body records and, last of them, the back_far record
            segments.append(("filler", streams[i].block(gaps[i])))
            segments.append(("record", lines[i]))
        segments.append(("filler", streams[n - 1].block(tails[0])))
        segments.append(("record", lines[n - 1]))                 # back_near record
        segments.append(("filler", streams[n].block(tails[1])))   # trailing filler: sets the near record's distance from the end
        prefix_ids, full_ids, full_text, prefix_text, char_spans = _measure(tok, cfg, segments, question)
        tspans = _token_spans(tok, prefix_text, char_spans)
        P = len(prefix_ids)
        dist = [P - e for _, e in tspans]
        err_len, err_near, err_far = target - P, d_near - dist[n - 1], d_far - dist[n - 2]
        history.append((P, dist[n - 1], dist[n - 2]))
        ok_len = abs(err_len) <= lay["length_tolerance_tokens"]
        ok_near = lay["back_near_tokens_from_end"][0] <= dist[n - 1] <= lay["back_near_tokens_from_end"][1]
        ok_far = lay["back_far_tokens_from_end"][0] <= dist[n - 2] <= lay["back_far_tokens_from_end"][1]
        if ok_len and ok_near and ok_far:
            return segments, {"prefix_ids": prefix_ids, "full_ids": full_ids, "full_text": full_text, "prefix_text": prefix_text, "token_spans": tspans,
                              "tokens_per_record": per, "layout_attempts": attempt + 1, "d_near_planned": d_near, "d_far_planned": d_far}
        if not ok_near:
            tails[1] = max(1, tails[1] + err_near)
        if not ok_far:
            tails[0] = max(1, tails[0] + err_far - (0 if ok_near else err_near))
        if not ok_len:
            share = err_len // n_body
            gaps = [max(1, g + share) for g in gaps]
            gaps[0] = max(1, gaps[0] + err_len - share * n_body)
    raise ValueError(f"could not lay out the document within the configured tolerances; history (P, d_near, d_far) = {history}")


def assemble(tok, cfg: dict, *, task: str, length_tag: str, doc_index: int, variant: str, ids: list[str], values: list[str],
             target_indices: list[int], question: str, buckets, rng: random.Random, changed_record_index=None, parent_doc_id=None, note=None) -> dict:
    """Lay out, render, tokenise, check the prefix cut and compute record spans for a fully specified record list."""
    data = cfg["data"]
    cfg = dict(cfg, _length_tag=length_tag)
    segments, m = layout_document(tok, cfg, rng, ids, values, question)
    full_ids, prefix_ids = m["full_ids"], m["prefix_ids"]
    closed = full_ids[:len(prefix_ids)] == prefix_ids
    if not closed:
        raise ValueError(f"prefix cut is not token-closed for {task}-{length_tag}-{doc_index}-{variant}")
    spans = m["token_spans"]
    n = len(ids)
    targets = [values[i] for i in target_indices]
    target_spans = [spans[i] for i in target_indices]
    prefix_len = len(prefix_ids)
    doc = {
        "doc_id": doc_identifier(task, length_tag, doc_index, variant), "task": task, "length_tag": length_tag,
        "target_tokens": int(data["lengths"][length_tag]), "doc_index": doc_index, "variant": variant, "parent_doc_id": parent_doc_id,
        "buckets": buckets, "n_records": n, "ids": ids, "values": values, "target_indices": target_indices, "targets": targets,
        "target_ids": [ids[i] for i in target_indices], "target_spans": target_spans,
        "target_distance_from_prefix_end": [prefix_len - s_[1] for s_ in target_spans],
        "target_records_from_end": [n - 1 - i for i in target_indices],
        "changed_record_index": changed_record_index, "question": question, "prefix_text": m["prefix_text"], "prompt_text": m["full_text"],
        "prefix_ids": prefix_ids, "suffix_ids": full_ids[len(prefix_ids):], "prefix_token_count": prefix_len,
        "prompt_token_count": len(full_ids), "suffix_token_count": len(full_ids) - prefix_len, "prefix_closed": closed,
        "record_spans": spans, "record_distance_from_prefix_end": [prefix_len - s_[1] for s_ in spans], "prefix_sha256": sha256_text(m["prefix_text"]),
        "prompt_sha256": sha256_text(m["full_text"]), "generator_version": data["generator_version"], "tokens_per_record": m["tokens_per_record"],
        "layout": {"attempts": m["layout_attempts"], "d_near_planned": m["d_near_planned"], "d_far_planned": m["d_far_planned"]}, "note": note,
    }
    return doc


def pick_index(rng: random.Random, bucket: str, n: int, buckets_cfg: dict) -> int:
    """Record index for a bucket: front / middle by fraction of the body records; back_near / back_far are the two tail
    records (the last and the second-to-last) placed at controlled token distances from the end."""
    spec = buckets_cfg[bucket]
    if "fraction" in spec:
        lo_f, hi_f = spec["fraction"]
        body = n - 2
        lo, hi = int(lo_f * body), max(int(lo_f * body) + 1, int(hi_f * body))
        return rng.randrange(lo, hi)
    return n - 1 - int(spec["from_end"])


def build_document(tok, cfg: dict, task: str, length_tag: str, doc_index: int) -> dict:
    """Base document for one grid cell entry (deterministic in the config seed, task, length and index)."""
    data = cfg["data"]
    plan = data["bucket_plan"][task][doc_index]
    seed = (data["seed"], task, length_tag, doc_index)
    rng = _rng(*seed)
    n = int(data["n_records"])
    id_lo, id_hi = data["id_range"]
    v_lo, v_hi = data["value_range"]
    ids = [f"{x:04d}" for x in rng.sample(range(id_lo, id_hi + 1), n)]
    values = [f"{rng.randint(v_lo, v_hi):04d}" for _ in range(n)]
    buckets_cfg = data["buckets"]
    if task == "T1":
        bucket = plan
        t = pick_index(rng, bucket, n, buckets_cfg)
        target_indices = [t]
        question = data["question"]["T1"].format(id=ids[t])
        buckets = [bucket]
    elif task == "T2":
        b1, b2 = plan
        i1, i2 = pick_index(rng, b1, n, buckets_cfg), pick_index(rng, b2, n, buckets_cfg)
        if i1 == i2:
            raise ValueError("T2 targets coincide")
        order = sorted([i1, i2]) if doc_index % 2 == 0 else sorted([i1, i2], reverse=True)
        target_indices = order
        question = data["question"]["T2"].format(id1=ids[order[0]], id2=ids[order[1]])
        buckets = [b1, b2]
    else:
        raise ValueError(f"unknown task {task}")
    doc = assemble(tok, cfg, task=task, length_tag=length_tag, doc_index=doc_index, variant="base", ids=ids, values=values,
                   target_indices=target_indices, question=question, buckets=buckets, rng=_rng(*seed, "layout"))
    doc["seed"] = "/".join(str(p) for p in seed)
    return doc


def rewrite_document(tok, cfg: dict, base: dict, variant: str, *, which: int = 0) -> dict:
    """Counterfactual rewrite of a base document; the whole prompt is re-rendered and re-tokenised (no KV reuse).

    "value": only the queried record's value changes. "irrelevant": one non-target record at least
    `irrelevant_min_distance_records` away changes its value. "move": the queried record swaps places with the back_near
    tail record (same token layout, the queried line now ends 30–70 tokens before the end).
    """
    if variant not in VARIANTS or variant == "base":
        raise ValueError(f"variant must be one of {VARIANTS[1:]}")
    data = cfg["data"]
    rng = _rng(base["seed"], variant, which)
    ids, values = list(base["ids"]), list(base["values"])
    tidx = list(base["target_indices"])
    n = len(ids)
    v_lo, v_hi = data["value_range"]
    changed, note = None, None
    if variant == "value":
        t = tidx[which]
        new = values[t]
        while new == values[t]:
            new = f"{rng.randint(v_lo, v_hi):04d}"
        values[t] = new
        changed = t
        note = f"value of queried record {which} changed {base['values'][t]} -> {new}"
    elif variant == "irrelevant":
        far = [i for i in range(n) if all(abs(i - t) >= data["irrelevant_min_distance_records"] for t in tidx)]
        j = rng.choice(far)
        new = values[j]
        while new == values[j]:
            new = f"{rng.randint(v_lo, v_hi):04d}"
        values[j] = new
        changed = j
        note = f"value of non-target record {j} changed {base['values'][j]} -> {new}"
    elif variant == "move":
        t = tidx[which]
        dest = n - 1 - int(data["buckets"]["back_near"]["from_end"])
        if t == dest:
            raise ValueError("the queried record already sits in the back_near slot")
        ids[t], ids[dest] = ids[dest], ids[t]
        values[t], values[dest] = values[dest], values[t]
        changed = dest
        note = f"queried record {which} swapped from index {t} into the back_near slot {dest}"
    id_to_index = {rid: i for i, rid in enumerate(ids)}
    new_tidx = [id_to_index[rid] for rid in base["target_ids"]]
    doc = assemble(tok, cfg, task=base["task"], length_tag=base["length_tag"], doc_index=base["doc_index"], variant=variant,
                   ids=ids, values=values, target_indices=new_tidx, question=base["question"], buckets=base["buckets"],
                   rng=_rng(*base["seed"].split("/"), "layout"), changed_record_index=changed, parent_doc_id=base["doc_id"], note=note)
    doc["seed"] = base["seed"]
    doc["rewrite_which"] = which
    if doc["prefix_token_count"] != base["prefix_token_count"]:
        raise ValueError("a rewrite changed the prefix token count")
    return doc


def data_digest(docs: list[dict]) -> str:
    """SHA-256 over the documents' prompt hashes and targets, in order (the data identity recorded in manifests)."""
    h = hashlib.sha256()
    for d in docs:
        h.update(json.dumps({"doc_id": d["doc_id"], "prompt_sha256": d["prompt_sha256"], "targets": d["targets"]}, ensure_ascii=False).encode("utf-8"))
    return h.hexdigest()
