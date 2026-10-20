"""Download real public splits and freeze 100 distinct conversations per dataset."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from core import choose_demos, norm, select_cases, stable_hash


def qrecc_case(r: dict) -> dict | None:
    if str(r.get('Conversation_source', '')).lower() != 'nq':
        # NQ avoids the QuAC overlap with CANARD.
        return None
    q, rw, answer = (str(r.get(k, '') or '').strip()
                      for k in ('Question','Truth_rewrite','Truth_answer'))
    if not q or not rw or not answer or not r.get('Context'):
        return None
    if int(r.get('Turn_no',0)) < 2:
        return None
    conv = str(r['Conversation_no'])
    return {'dataset':'qrecc', 'id':f'qrecc:{conv}:{r["Turn_no"]}', 'group':f'qrecc:{conv}',
            'history':[str(s) for s in r['Context'] if s], 'question':q,
            'gold_rewrite':rw, 'answer':answer,
            'gold_passages':[str(x) for x in (r.get('Truth_passages') or []) if str(x)]}


def canard_case(r: dict) -> dict | None:
    q, rw, answer, evidence = (str(r.get(k,'') or '').strip()
                               for k in ('Question','Rewrite','answer','true_contexts'))
    if not all([q,rw,answer,evidence]):
        return None
    if norm(answer) in {'cannotanswer','unknown','i dont know','not sure'}:
        return None
    # Keep answerable samples to make extractive EM meaningful.
    if norm(answer) not in norm(evidence):
        return None
    group = str(r.get('QuAC_dialog_id',''))
    if not group:
        return None
    return {'dataset':'canard', 'id':f'canard:{group}:{r.get("Question_no",0)}',
            'group':f'canard:{group}', 'history':[str(s) for s in (r.get('History') or []) if s],
            'question':q, 'gold_rewrite':rw, 'answer':answer,
            'evidence':evidence, 'distractors':[str(s) for s in (r.get('retrieved_contexts') or []) if s]}


def build(n=100, shots=3, seed=42, out_dir='data'):
    from datasets import load_dataset
    out=Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    config = [
        ('qrecc','slupart/qrecc',qrecc_case, 'nq_only'),
        ('canard','gaussalgo/Canard_Wiki-augmented',canard_case,'extractive_only'),
    ]
    prepared = {}
    for offset,(name,repo,normalizer,filter_name) in enumerate(config):
        print(f'[prepare] loading {repo}: train/test ...',flush=True)
        tr = [c for r in load_dataset(repo,split='train') if (c:=normalizer(r))]
        te = [c for r in load_dataset(repo,split='test') if (c:=normalizer(r))]
        evals = select_cases(te, n, seed+offset*11)
        demos = choose_demos(tr, shots, seed+1000+offset, evals)
        if len(evals)!=n or len(demos)!=shots:
            raise AssertionError('Dataset size or number of demos not met')
        if {x['group'] for x in demos} & {x['group'] for x in evals}:
            raise AssertionError('Train / test conversation leakage')
        prepared[name]={'source':repo, 'filter':filter_name,'eval':evals,'demos':demos,
                        'n':len(evals), 'num_shots':len(demos)}
        (out/f'{name}.json').write_text(json.dumps(prepared[name],ensure_ascii=False,indent=2),encoding='utf-8')
        print(f'  {name}: {len(evals)} eval / {len(demos)} demos / eligible train={len(tr)} test={len(te)}',flush=True)
    # Extra defense against exact cross-dataset query/rewrite collisions.
    for target,other in [('qrecc','canard'),('canard','qrecc')]:
        ev = prepared[target]['eval']
        ex = prepared[other]['demos']
        blocked = {norm(c['question']) for c in ev} | {norm(c['gold_rewrite']) for c in ev}
        if any(norm(c['question']) in blocked or norm(c['gold_rewrite']) in blocked for c in ex):
            raise AssertionError('Cross-domain demonstration overlaps test text')
    manifest={'seed':seed,'n_per_dataset':n,'shots':shots,'datasets':{
        k:{'source':v['source'],'filter':v['filter'], 'eval_ids':[x['id'] for x in v['eval']],
           'demo_ids':[x['id'] for x in v['demos']]} for k,v in prepared.items()}}
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print('[prepare] frozen manifest and datasets written to:',out.resolve())


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--n',type=int,default=100)
    p.add_argument('--shots',type=int,default=3);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--out-dir',default='data')
    a=p.parse_args();build(a.n,a.shots,a.seed,a.out_dir)
