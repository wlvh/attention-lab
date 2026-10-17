"""G2-lite task data: programmatically generated record tables with a known answer, built from the frozen config.

A document is a header plus `n` lines "编号 XXXX：YYYY" (one record per line; every record is exactly 12 Qwen3 tokens
because the tokenizer splits digits one per token). Ids are distinct four-digit numbers, values are random four-digit
numbers, both bound afresh per document from a seed derived from (data seed, task, length, document index). The
question is appended after the table in the same user turn; the chat template (thinking disabled) wraps everything.

The common prefix is the rendered text before the question marker. It is tokenised on its own and checked to be an
exact token prefix of the full prompt, so every read path starts from identical prefix tokens and the question never
enters the shared prefix (docs/PLAN_CURRENT.md §8, docs/tasks/05_G2_LITE.md).

Target positions are chosen by bucket: "front" / "middle" by record-index fraction, "back_near" / "back_far" by record
index from the end (3–5 and 13–20 records, i.e. about 36–60 and 156–240 tokens before the end of the table), so that
the recent-window read B covers back_near at every rho and back_far only when k is large enough. T2 asks for two
far-apart records in a stated order (ascending position for even document indices, descending for odd ones).

Rewrites (counterfactuals) rebuild the whole document: "value" changes only the queried record's value; "irrelevant"
changes only one non-target record at least 10 records away from every target; "move" moves one queried record to the
back_near region (index from end 3) and leaves every other record in order. Nothing here loads the model.
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


def render_prompt(tok, content_prefix: str, question: str, *, enable_thinking: bool = False) -> tuple[str, str]:
    """Full chat-template text (generation prompt appended, thinking disabled) and the text before the question."""
    msgs = [{"role": "user", "content": content_prefix + question}]
    full = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking)
    i = full.rfind(question)
    if i < 0 or not question.startswith(QUESTION_MARKER):
        raise ValueError("the question must start with the marker and appear once at the end of the user turn")
    return full, full[:i]


def encode(tok, text: str) -> list[int]:
    return list(tok(text, add_special_tokens=False).input_ids)


def pick_index(rng: random.Random, bucket: str, n: int, buckets_cfg: dict) -> int:
    spec = buckets_cfg[bucket]
    if "fraction" in spec:
        lo_f, hi_f = spec["fraction"]
        lo, hi = int(lo_f * n), int(hi_f * n)
        if hi <= lo:
            raise ValueError(f"bucket {bucket} empty for n={n}")
        return rng.randrange(lo, hi)
    lo_e, hi_e = spec["from_end"]                       # inclusive range of "records from the end"
    return n - 1 - rng.randint(lo_e, hi_e)


def record_count_for(tok, cfg: dict, target_tokens: int) -> tuple[int, int]:
    """n records whose prefix comes closest to `target_tokens`, and the tokens per record actually observed."""
    data = cfg["data"]
    header, fmt = data["header"], data["record_format"]
    probe_q = data["question"]["T1"].format(id="1234")

    def prefix_len(n):
        lines = "".join(fmt.format(id=f"{1000 + i:04d}", value="5678") for i in range(n))
        _, pre = render_prompt(tok, header + lines + data["table_end"], probe_q)
        return len(encode(tok, pre))

    len8, len9, len16 = prefix_len(8), prefix_len(9), prefix_len(16)
    per = len9 - len8                                   # measured between two non-empty tables (the table end merges differently when empty)
    if len16 - len8 != 8 * per:
        raise ValueError(f"records do not tokenise to a constant length ({len16 - len8} tokens for 8 records vs {per} each)")
    base = len8 - 8 * per                               # tokens that are not records: template head, header, table end
    n = max(1, round((target_tokens - base) / per))
    best = min((abs(prefix_len(m) - target_tokens), m) for m in (n - 1, n, n + 1) if m >= 1)[1]
    return best, per


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


def assemble(tok, cfg: dict, *, task: str, length_tag: str, doc_index: int, variant: str, ids: list[str], values: list[str],
             target_indices: list[int], question: str, buckets, changed_record_index=None, parent_doc_id=None, note=None) -> dict:
    """Render, tokenise, check the prefix cut and compute record spans for a fully specified record table."""
    data = cfg["data"]
    header, fmt, table_end = data["header"], data["record_format"], data["table_end"]
    lines = [fmt.format(id=i, value=v) for i, v in zip(ids, values)]
    content = header + "".join(lines) + table_end
    full_text, prefix_text = render_prompt(tok, content, question, enable_thinking=bool(cfg["prompt"]["enable_thinking"]))
    full_ids, prefix_ids = encode(tok, full_text), encode(tok, prefix_text)
    closed = full_ids[:len(prefix_ids)] == prefix_ids
    if not closed:
        raise ValueError(f"prefix cut is not token-closed for {task}-{length_tag}-{doc_index}-{variant}")
    head = full_text.index(header) + len(header)
    char_spans, pos = [], head
    for line in lines:
        char_spans.append((pos, pos + len(line.rstrip("\n"))))
        pos += len(line)
    spans = _token_spans(tok, prefix_text, char_spans)
    n = len(ids)
    targets = [values[i] for i in target_indices]
    target_spans = [spans[i] for i in target_indices]
    prefix_len = len(prefix_ids)
    doc = {
        "doc_id": doc_identifier(task, length_tag, doc_index, variant), "task": task, "length_tag": length_tag,
        "target_tokens": int(data["lengths"][length_tag]), "doc_index": doc_index, "variant": variant, "parent_doc_id": parent_doc_id,
        "buckets": buckets, "n_records": n, "ids": ids, "values": values, "target_indices": target_indices, "targets": targets,
        "target_ids": [ids[i] for i in target_indices], "target_spans": target_spans,
        "target_distance_from_prefix_end": [prefix_len - s[1] for s in target_spans],
        "target_records_from_end": [n - 1 - i for i in target_indices],
        "changed_record_index": changed_record_index, "question": question, "prefix_text": prefix_text, "prompt_text": full_text,
        "prefix_ids": prefix_ids, "suffix_ids": full_ids[len(prefix_ids):], "prefix_token_count": prefix_len,
        "prompt_token_count": len(full_ids), "suffix_token_count": len(full_ids) - prefix_len, "prefix_closed": closed,
        "record_spans": spans, "prefix_sha256": sha256_text(prefix_text), "prompt_sha256": sha256_text(full_text),
        "generator_version": data["generator_version"], "note": note,
    }
    return doc


def build_document(tok, cfg: dict, task: str, length_tag: str, doc_index: int) -> dict:
    """Base document for one grid cell entry (deterministic in the config seed, task, length and index)."""
    data = cfg["data"]
    plan = data["bucket_plan"][task][doc_index]
    seed = (data["seed"], task, length_tag, doc_index)
    rng = _rng(*seed)
    n, per = record_count_for(tok, cfg, int(data["lengths"][length_tag]))
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
                   target_indices=target_indices, question=question, buckets=buckets)
    doc["tokens_per_record"] = per
    doc["seed"] = "/".join(str(p) for p in seed)
    return doc


def rewrite_document(tok, cfg: dict, base: dict, variant: str, *, which: int = 0) -> dict:
    """Counterfactual rewrite of a base document; the whole prompt is re-rendered and re-tokenised (no KV reuse)."""
    if variant not in VARIANTS or variant == "base":
        raise ValueError(f"variant must be one of {VARIANTS[1:]}")
    data = cfg["data"]
    rng = _rng(base["seed"], variant, which)
    ids, values = list(base["ids"]), list(base["values"])
    tidx = list(base["target_indices"])
    n = len(ids)
    v_lo, v_hi = data["value_range"]
    changed = None
    note = None
    if variant == "value":                       # only the queried record's value changes -> the answer must follow
        t = tidx[which]
        new = values[t]
        while new == values[t]:
            new = f"{rng.randint(v_lo, v_hi):04d}"
        values[t] = new
        changed = t
        note = f"value of queried record {which} changed {base['values'][t]} -> {new}"
    elif variant == "irrelevant":                # one far-away non-target record changes -> the answer must not move
        far = [i for i in range(n) if all(abs(i - t) >= data["irrelevant_min_distance_records"] for t in tidx)]
        j = rng.choice(far)
        new = values[j]
        while new == values[j]:
            new = f"{rng.randint(v_lo, v_hi):04d}"
        values[j] = new
        changed = j
        note = f"value of non-target record {j} changed {base['values'][j]} -> {new}"
    elif variant == "move":                      # the queried record moves into the back_near region (3 from the end)
        t = tidx[which]
        rid, rval = ids.pop(t), values.pop(t)
        dest = n - 1 - data["move_to_records_from_end"]
        ids.insert(dest, rid)
        values.insert(dest, rval)
        changed = dest
        note = f"queried record {which} moved from index {t} to {dest} (records from end {data['move_to_records_from_end']})"
    id_to_index = {rid: i for i, rid in enumerate(ids)}
    new_tidx = [id_to_index[rid] for rid in base["target_ids"]]
    doc = assemble(tok, cfg, task=base["task"], length_tag=base["length_tag"], doc_index=base["doc_index"], variant=variant,
                   ids=ids, values=values, target_indices=new_tidx, question=base["question"], buckets=base["buckets"],
                   changed_record_index=changed, parent_doc_id=base["doc_id"], note=note)
    doc["seed"] = base["seed"]
    doc["tokens_per_record"] = base.get("tokens_per_record")
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
