"""Pure (API-independent) benchmark helpers."""
from __future__ import annotations
import hashlib
import json
import random
import re
import string
from collections import Counter
from typing import Any

MODES = ('zero_shot', 'same_3shot', 'cross_3shot', 'generic_3shot')
DATASETS = ('qrecc', 'canard')
SYSTEM_REWRITE = (
    'You rewrite conversational search questions. Resolve references and ellipsis '
    'using only the conversation history. Preserve all names, numbers, constraints '
    'and intent. Do not answer the question. Do not invent facts or add unnecessary '
    'details. Return only one standalone English search question, without quotes.'
)
SYSTEM_ANSWER = (
    'Answer the question using only the supplied retrieved passages. '
    'Be concise and factual. If the evidence is insufficient, respond exactly '
    'CANNOTANSWER. Do not use outside knowledge.'
)


def norm(s: str) -> str:
    s = s.lower()
    s = ''.join(c for c in s if c not in string.punctuation)
    return ' '.join(w for w in s.split() if w not in {'a','an','the'})


def token_f1(pred: str, target: str) -> float:
    p, t = norm(pred).split(), norm(target).split()
    if not p or not t:
        return float(p == t)
    n = sum((Counter(p) & Counter(t)).values())
    return (2 * n / (len(p) + len(t))) if n else 0.0


def score_answer(pred: str, gold: str) -> dict:
    return {'em': float(norm(pred) == norm(gold)), 'f1': token_f1(pred, gold)}


def serialize_case(case: dict) -> str:
    turns = case['history']
    hist = '\n'.join(f'{i+1}. {t}' for i,t in enumerate(turns)) if turns else '(none)'
    return f'Conversation history (oldest to newest):\n{hist}\nCurrent question: {case["question"]}'


def validate_generic_demos(demos: list[dict], forbidden: list[dict] | None = None) -> None:
    """Validate independent hand-authored examples and exact-text eval separation."""
    if len(demos) != 3:
        raise ValueError(f'Generic condition must have exactly 3 demonstrations; got {len(demos)}')
    eval_texts = {norm(c[k]) for c in (forbidden or []) for k in ('question', 'gold_rewrite')}
    seen = set()
    for i, demo in enumerate(demos):
        if not isinstance(demo.get('history'), list) or not demo['history']:
            raise ValueError(f'Generic example {i} requires nonempty history list')
        if not all(isinstance(x, str) and x.strip() for x in demo['history']):
            raise ValueError(f'Generic example {i} has invalid history text')
        if not all(isinstance(demo.get(k), str) and demo[k].strip() for k in ('question', 'gold_rewrite')):
            raise ValueError(f'Generic example {i} requires question and gold_rewrite')
        for k in ('question', 'gold_rewrite'):
            normalized = norm(demo[k])
            if normalized in eval_texts:
                raise ValueError(f'Generic example {i} matches evaluation {k} text')
            if normalized in seen:
                raise ValueError(f'Generic example {i} duplicates a prior question/rewrite')
            seen.add(normalized)


def rewrite_prompt(case: dict, demos: list[dict]) -> str:
    # Exactly the same instructions for every condition; demonstrations are the only change.
    prefix = ''
    if demos:
        blocks = []
        for i, ex in enumerate(demos, 1):
            blocks.append(f'Example {i}:\n{serialize_case(ex)}\nStandalone rewrite: {ex["gold_rewrite"]}')
        prefix = '\n\n'.join(blocks) + '\n\n---\n\n'
    return prefix + serialize_case(case) + '\nStandalone rewrite:'


def stable_hash(s: str) -> str:
    return hashlib.sha256(s.encode('utf-8')).hexdigest()[:16]


def select_cases(rows: list[dict], n: int, seed: int, *, excluded: set[str] | None = None) -> list[dict]:
    rng = random.Random(seed)
    choices = list(rows)
    rng.shuffle(choices)
    used_groups, used_questions = set(), set(excluded or ())
    out = []
    for c in choices:
        q_sig = norm(c['question'] + ' || ' + c['gold_rewrite'])
        if c['group'] in used_groups or q_sig in used_questions:
            continue
        out.append(c)
        used_groups.add(c['group'])
        used_questions.add(q_sig)
        if len(out) == n:
            return out
    raise ValueError(f'Wanted {n} distinct conversations, got {len(out)}')


def choose_demos(train: list[dict], n: int, seed: int, forbidden: list[dict]) -> list[dict]:
    # Prefer non-trivial (genuinely context-dependent) rewrites; no eval overlap.
    blocked = {norm(c['question']) for c in forbidden}
    blocked_rewrites = {norm(c['gold_rewrite']) for c in forbidden}
    candidates = [c for c in train if norm(c['question']) != norm(c['gold_rewrite'])
                  and norm(c['question']) not in blocked
                  and norm(c['gold_rewrite']) not in blocked_rewrites]
    return select_cases(candidates, n, seed)


def make_doc(doc_id: str, text: str, source: str) -> dict:
    return {'id': str(doc_id), 'text': ' '.join(str(text).split())[:3600], 'source': source}


def build_corpus(ds: str, cases: list[dict], qrecc_mode='proxy', external_docs=None) -> tuple[list[dict], dict[str,set[str]], str]:
    docs: dict[str, dict] = {}
    qrels: dict[str, set[str]] = {}
    if ds == 'canard':
        for c in cases:
            pos = 'canard:' + stable_hash(c['evidence'])
            # Gold-containing fixed chunk. Selecting the window uses the gold answer
            # ONLY at corpus-construction time, never in rewrites or reader prompts.
            evidence = c['evidence']
            match = evidence.lower().find(c['answer'].lower())
            if match >= 0 and len(evidence) > 3600:
                start = max(0, match - 1600)
                start = min(start, max(0,len(evidence)-3600))
                evidence = evidence[start:start+3600]
            docs[pos] = make_doc(pos, evidence, 'oracle_positive_chunk')
            qrels[c['id']] = {pos}
            for neg in c.get('distractors', [])[:3]:
                if not neg or len(neg.strip()) < 60:
                    continue
                nid = 'canard:' + stable_hash(neg)
                docs.setdefault(nid, make_doc(nid, neg, 'provided_distractor'))
        label = 'closed_pool_gold_context_plus_provided_distractors'
    elif qrecc_mode == 'external':
        if not external_docs:
            raise ValueError('QReCC external mode requires --qrecc-docs JSONL with id,text')
        docs = {str(d['id']): make_doc(str(d['id']), d['text'], 'external_passage') for d in external_docs}
        for c in cases:
            gold_ids = {str(pid) for pid in c['gold_passages']}
            present = gold_ids & docs.keys()
            if not present:
                raise ValueError(f'No gold passage for QReCC case {c["id"]} in supplied corpus')
            qrels[c['id']] = present
        label = 'provided_external_passage_corpus'
    elif qrecc_mode == 'proxy':
        # Deliberately NOT official retrieval. The answer is reused as a synthetic
        # "document"; top-k ranking and downstream EM here can be artificially high.
        for c in cases:
            pos = 'answer-proxy:' + stable_hash(c['answer'])
            docs[pos] = make_doc(pos, c['answer'], 'gold_answer_as_document_NOT_EVIDENCE')
            qrels[c['id']] = {pos}
        label = 'ANSWER_AS_DOCUMENT_PROXY_NOT_OFFICIAL_RETRIEVAL'
    elif qrecc_mode == 'rewrite_only':
        return [], {}, 'rewrite_only'
    else:
        raise ValueError(qrecc_mode)
    return list(docs.values()), qrels, label


def retrieval_scores(doc_ids: list[str], relevant: set[str], k=10) -> dict:
    def hit_at(n): return float(bool(set(doc_ids[:n]) & relevant))
    rank = next((i+1 for i,d in enumerate(doc_ids[:k]) if d in relevant), None)
    # Binary relevance; denominator is ideal DCG for the relevant docs.
    import math
    dcg = sum(1/math.log2(i+2) for i,d in enumerate(doc_ids[:k]) if d in relevant)
    ideal = sum(1/math.log2(i+2) for i in range(min(len(relevant), k)))
    return {'recall_at_5': len(set(doc_ids[:5]) & relevant)/len(relevant) if relevant else 0.,
            'hit_at_5': hit_at(5),
            'mrr_at_10': 1/rank if rank else 0.,
            'ndcg_at_10': dcg/ideal if ideal else 0.}


def parse_response(response: Any) -> str:
    result = getattr(response, 'output_text', '')
    if not result:
        raise RuntimeError('OpenAI returned no text output')
    return result.strip().strip('`').strip().strip('"')


def answer_prompt(question: str, docs: list[dict]) -> str:
    lines = [f'Question: {question}', 'Retrieved passages:']
    lines += [f'[{idx}] {d["text"]}' for idx, d in enumerate(docs,1)]
    lines.append('Answer:')
    return '\n\n'.join(lines)


def bootstrap_diff(rows_a: list[dict], rows_b: list[dict], metric: str, seed=42, n_boot=1500) -> dict:
    """Paired bootstrap CI of (A-B), on exactly matching case IDs."""
    aa = {x['case_id']:x for x in rows_a}
    bb = {x['case_id']:x for x in rows_b}
    common = sorted(aa.keys() & bb.keys())
    # Do not silently treat missing API responses as successes or zeros.
    pairs = [(float(aa[k][metric]), float(bb[k][metric])) for k in common
             if aa[k].get(metric) is not None and bb[k].get(metric) is not None]
    if not pairs:
        return {'n':0, 'delta':None, 'ci95':None}
    import numpy as np
    differences = np.array([a-b for a,b in pairs],dtype=float)
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0,len(differences),(n_boot,len(differences)))
    vals = differences[sampled].mean(axis=1)
    return {'n':len(pairs), 'delta':round(float(differences.mean()),5),
            'ci95':[round(float(np.quantile(vals,.025)),5),round(float(np.quantile(vals,.975)),5)]}
