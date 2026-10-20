"""Run GPT-6 Luna query-rewriting experiment, fixed retrieval and QA.

The `proxy` QReCC evidence mode is NOT official passage retrieval.
"""
from __future__ import annotations
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import time
from collections import defaultdict
from core import (MODES, DATASETS, SYSTEM_REWRITE, SYSTEM_ANSWER, answer_prompt,
                  bootstrap_diff, build_corpus, parse_response, retrieval_scores,
                  rewrite_prompt, score_answer, validate_generic_demos, stable_hash)


def read_jsonl(path):
    with open(path,encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


class GPTClient:
    def __init__(self, model, cache_dir):
        if not os.getenv('OPENAI_API_KEY'):
            raise EnvironmentError('Set OPENAI_API_KEY (never commit it or put it in the repo).')
        from openai import OpenAI
        self.model=model
        self.client=OpenAI(timeout=90.0,max_retries=2)
        self.cache=Path(cache_dir)
        self.cache.mkdir(parents=True,exist_ok=True)

    def ask(self, task, system, prompt, max_tokens):
        key=hashlib.sha256(json.dumps([self.model,task,system,prompt,max_tokens],
            ensure_ascii=False).encode()).hexdigest()
        cached=self.cache/(key+'.json')
        if cached.exists():
            return json.loads(cached.read_text(encoding='utf-8'))
        start=time.perf_counter()
        # Luna supports reasoning.effort='none'. Keep model settings identical.
        response=self.client.responses.create(model=self.model,instructions=system,input=prompt,
            max_output_tokens=max_tokens,reasoning={'effort':'none'},store=False)
        result={'text':parse_response(response),
                'latency_ms':round(1000*(time.perf_counter()-start),2),
                'input_tokens':getattr(getattr(response,'usage',None),'input_tokens',None),
                'output_tokens':getattr(getattr(response,'usage',None),'output_tokens',None),
                'cached':False}
        temp=cached.with_suffix('.tmp.'+str(os.getpid())+'.'+str(time.thread_time_ns()))
        temp.write_text(json.dumps(result,ensure_ascii=False),encoding='utf-8')
        os.replace(temp,cached)
        return result


class HybridRetriever:
    def __init__(self, docs, dense_name, cross_name, top_candidates=30):
        import numpy as np
        from rank_bm25 import BM25Okapi
        from sentence_transformers import SentenceTransformer, CrossEncoder
        self.docs=docs
        self.top_candidates=top_candidates
        self.np=np
        self.tokenized=[self.tokenize(d['text']) for d in docs]
        self.bm25=BM25Okapi(self.tokenized)
        self.encoder=SentenceTransformer(dense_name)
        self.cross=CrossEncoder(cross_name)
        texts=[d['text'] for d in docs]
        self.emb=self.encoder.encode(texts,batch_size=32,normalize_embeddings=True,
                                     show_progress_bar=True,convert_to_numpy=True)

    @staticmethod
    def tokenize(text):
        import re
        return re.findall(r"[a-z0-9]+",text.lower())

    def search(self, query, k=10):
        np=self.np
        start=time.perf_counter()
        bm_scores=self.bm25.get_scores(self.tokenize(query))
        qemb=self.encoder.encode([query],normalize_embeddings=True,convert_to_numpy=True)[0]
        dense_scores=self.emb @ qemb
        n=min(self.top_candidates,len(self.docs))
        bm_ids=np.argsort(-bm_scores,kind='stable')[:n].tolist()
        de_ids=np.argsort(-dense_scores,kind='stable')[:n].tolist()
        rrf=defaultdict(float)
        for rank, idx in enumerate(bm_ids,1): rrf[idx]+=1/(60+rank)
        for rank, idx in enumerate(de_ids,1): rrf[idx]+=1/(60+rank)
        candidates=sorted(rrf,key=lambda i:(-rrf[i],i))[:n]
        pairs=[(query,self.docs[i]['text']) for i in candidates]
        scores=self.cross.predict(pairs,batch_size=32,show_progress_bar=False)
        ranked=sorted(zip(candidates,scores),key=lambda t:(-float(t[1]),t[0]))
        ids=[idx for idx,_ in ranked[:k]]
        return [self.docs[idx] for idx in ids], round((time.perf_counter()-start)*1000,2)


def summary_and_stats(records, out_dir, model, generic_demo_fingerprint=None):
    import numpy as np
    metrics=('rewrite_em','rewrite_f1','recall_at_5','hit_at_5','mrr_at_10',
             'ndcg_at_10','answer_em','answer_f1','rewrite_latency_ms',
             'retrieval_latency_ms','answer_latency_ms','rewrite_input_tokens','rewrite_output_tokens','answer_input_tokens','answer_output_tokens','total_latency_ms')
    groups=defaultdict(list)
    for r in records:
        groups[(r['dataset'],r['condition'])].append(r)
    report={'model':model,'generic_demo_fingerprint':generic_demo_fingerprint,'groups':{},'paired_differences':{},
            'warnings':['QReCC proxy scores use gold ANSWERS as synthetic evidence and are NOT official retrieval / end-to-end QA benchmarks.',
                        'CANARD closed-pool retrieval includes test gold contexts and dataset-provided distractors; not full-corpus Wikipedia evaluation.',
                        'Unlabeled relevant distractors may exist; top-k qrels are incomplete.',
                        'No-rewrite has 0 rewrite API latency; comparisons of total latency should include answer calls.',
                        'Generic few-shot uses 3 fixed hand-authored synthetic examples, identical for both datasets; example content/length may confound differences.']}
    for (dataset,condition),items in sorted(groups.items()):
        vals={}
        for metric in metrics:
            num=[float(x[metric]) for x in items if x.get(metric) is not None]
            vals[metric]=round(float(np.mean(num)),5) if num else None
        vals['evaluated_cases']=len(items)
        vals['rewrite_completed']=sum(x.get('rewrite_em') is not None for x in items)
        vals['answers_completed']=sum(x.get('answer_em') is not None for x in items)
        vals['retrieval_completed']=sum(x.get('mrr_at_10') is not None for x in items)
        vals['retrieval_protocol']=items[0]['retrieval_protocol'] if items else None
        vals['rewrite_errors']=sum(bool(x.get('rewrite_error')) for x in items)
        vals['answer_errors']=sum(bool(x.get('answer_error')) for x in items)
        report['groups'][f'{dataset}/{condition}']=vals
    for ds in DATASETS:
        base=groups.get((ds,'zero_shot'),[])
        for condition in MODES[1:]:
            rows=groups.get((ds,condition),[])
            report['paired_differences'][f'{ds}/{condition}_minus_zero_shot']={
                metric:bootstrap_diff(rows,base,metric) for metric in metrics
                if metric in ('rewrite_em','rewrite_f1','recall_at_5','mrr_at_10',
                              'ndcg_at_10','answer_em','answer_f1','rewrite_latency_ms')}
    # Case-level paired win/loss/tie rates (same exact input examples).
    for ds in DATASETS:
        zero={x['case_id']:x for x in groups.get((ds,'zero_shot'),[])}
        for mode in MODES[1:]:
            candidate={x['case_id']:x for x in groups.get((ds,mode),[])}
            for metric in ('rewrite_f1','mrr_at_10','answer_em'):
                joined=[(candidate[i][metric],zero[i][metric]) for i in candidate.keys() & zero.keys()
                        if candidate[i].get(metric) is not None and zero[i].get(metric) is not None]
                wins=sum(a>b for a,b in joined);losses=sum(a<b for a,b in joined)
                report.setdefault('case_level_comparison',{})[f'{ds}/{mode}/{metric}']={
                    'n':len(joined),'wins':wins,'losses':losses,'ties':len(joined)-wins-losses}
    out=Path(out_dir)
    (out/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('\n=== SUMMARY (higher is better except latency) ===')
    for key,r in report['groups'].items():
        print(f'{key:22} rewrite_F1={r["rewrite_f1"]}  MRR@10={r["mrr_at_10"]}  Answer_EM={r["answer_em"]}  coverage={r["rewrite_completed"]}/{r["evaluated_cases"]}')
    import csv
    with (out/'summary.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f)
        writer.writerow(['dataset','condition','rewrite_f1','recall_at_5','mrr_at_10','ndcg_at_10','answer_em','answer_f1','rewrite_latency_ms','total_latency_ms','protocol'])
        for key, r in report['groups'].items():
            ds, mode = key.split('/')
            writer.writerow([ds,mode]+[r.get(m) for m in ('rewrite_f1','recall_at_5','mrr_at_10','ndcg_at_10','answer_em','answer_f1','rewrite_latency_ms','total_latency_ms','retrieval_protocol')])
    print('\nIMPORTANT: QReCC proxy retrieval/QA is synthetic and NOT comparable to official QReCC full-corpus metrics.')
    print('Result files:',(out/'summary.json').resolve())
    return report


def run(args):
    data=Path(args.data_dir)
    prepared={name:json.loads((data/f'{name}.json').read_text(encoding='utf-8')) for name in DATASETS}
    if any(len(p['eval'])!=args.expected_n for p in prepared.values()):
        raise ValueError(f'Expected exactly {args.expected_n} frozen evaluation samples per dataset; rerun prepare.py')
    if any(len(p['demos'])!=3 for p in prepared.values()):
        raise ValueError('This is a controlled 3-shot experiment. Rerun prepare.py --shots 3.')
    generic_path=Path(args.generic_demos)
    generic_demos=json.loads(generic_path.read_text(encoding='utf-8'))
    validate_generic_demos(generic_demos, [c for item in prepared.values() for c in item['eval']])
    generic_fingerprint=stable_hash(json.dumps(generic_demos,sort_keys=True,ensure_ascii=False))
    print(f'[generic examples] {generic_path} (sha256 prefix={generic_fingerprint}, 3 hand-authored demos)')
    out=Path(args.out_dir);out.mkdir(parents=True,exist_ok=True)
    api=GPTClient(args.model,out/'api_cache')
    conditions=['raw_question']+list(MODES) if not args.no_raw else list(MODES)
    jobs=[]
    for ds in DATASETS:
        other='canard' if ds=='qrecc' else 'qrecc'
        for case in prepared[ds]['eval']:
            for mode in conditions:
                demos=(prepared[ds]['demos'] if mode=='same_3shot'
                       else prepared[other]['demos'] if mode=='cross_3shot'
                       else generic_demos if mode=='generic_3shot' else [])
                jobs.append((ds,mode,case,demos))
    print(f'[rewrite] {len(jobs)} total case-condition combinations, {sum(j[1] not in ("raw_question",) for j in jobs)} paid rewrites')

    def rewriting(job):
        ds,mode,case,demos=job
        if mode=='raw_question':
            return {'dataset':ds,'condition':mode,'case_id':case['id'],
                    'rewrite':case['question'],'rewrite_latency_ms':0.0,
                    'rewrite_input_tokens':0,'rewrite_output_tokens':0,
                    'rewrite_error':None}
        try:
            v=api.ask('rewrite',SYSTEM_REWRITE,rewrite_prompt(case,demos),max_tokens=180)
            return {'dataset':ds,'condition':mode,'case_id':case['id'],
                    'rewrite':v['text'].strip(), 'rewrite_latency_ms':v['latency_ms'],
                    'rewrite_input_tokens':v['input_tokens'],'rewrite_output_tokens':v['output_tokens'],
                    'rewrite_error':None}
        except Exception as exc:
            return {'dataset':ds,'condition':mode,'case_id':case['id'],
                    'rewrite':'','rewrite_latency_ms':None,
                    'rewrite_error':f'{type(exc).__name__}: {exc}'}

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        outputs=list(pool.map(rewriting,jobs))
    rewrite_lookup={(x['dataset'],x['condition'],x['case_id']):x for x in outputs}
    (out/'rewrites.jsonl').write_text('\n'.join(json.dumps(r,ensure_ascii=False) for r in outputs)+'\n',encoding='utf-8')
    failed=[x for x in outputs if x['rewrite_error']]
    if failed:
        print(f'[warning] {len(failed)} rewrite calls FAILED. Examples:',failed[:2])
    if args.abort_on_error and failed:
        raise RuntimeError('Incomplete rewriting; inspect rewrites.jsonl and API key/access. Retries use cache.')

    external_docs=read_jsonl(args.qrecc_docs) if args.qrecc_docs else None
    final=[]
    for ds in DATASETS:
        cases=prepared[ds]['eval']
        mode=args.qrecc_mode if ds=='qrecc' else 'gold'
        docs,qrels,protocol=build_corpus(ds,cases,qrecc_mode=mode,external_docs=external_docs)
        print(f'[{ds}] corpus={len(docs)}, protocol={protocol}')
        ranker=HybridRetriever(docs,args.dense_model,args.cross_encoder,args.candidates) if docs and not args.skip_retrieval else None
        for case in cases:
            for condition in conditions:
                obj=rewrite_lookup[(ds,condition,case['id'])]
                row={**obj,'gold_rewrite':case['gold_rewrite'],'gold_answer':case['answer'],
                     'question':case['question'],'retrieval_protocol':protocol,
                     'rewrite_em':None,'rewrite_f1':None,
                     'recall_at_5':None,'hit_at_5':None,'mrr_at_10':None,'ndcg_at_10':None,
                     'retrieval_latency_ms':None,'answer_em':None,'answer_f1':None,
                     'answer_latency_ms':None,'answer_input_tokens':None,'answer_output_tokens':None,'total_latency_ms':None,'answer_error':None}
                if not obj['rewrite_error']:
                    rewrite=obj['rewrite']
                    rewrite_score=score_answer(rewrite,case['gold_rewrite'])
                    row.update({'rewrite_em':rewrite_score['em'],'rewrite_f1':rewrite_score['f1']})
                    if ranker:
                        found,t=ranker.search(rewrite,k=10)
                        row['retrieved_ids']=[d['id'] for d in found]
                        row['retrieval_latency_ms']=t
                        row.update(retrieval_scores(row['retrieved_ids'],qrels[case['id']]))
                        if not args.skip_answer:
                            try:
                                # IMPORTANT: no target answer or target rewrite fed to reader.
                                answer=api.ask('answer',SYSTEM_ANSWER,answer_prompt(rewrite,found[:5]),max_tokens=160)
                                row['predicted_answer']=answer['text']
                                row.update({'answer_em':score_answer(answer['text'],case['answer'])['em'],
                                            'answer_f1':score_answer(answer['text'],case['answer'])['f1'],
                                            'answer_latency_ms':answer['latency_ms'],
                                            'answer_input_tokens':answer['input_tokens'],
                                            'answer_output_tokens':answer['output_tokens']})
                            except Exception as exc:
                                row['answer_error']=f'{type(exc).__name__}: {exc}'
                times=[row.get(k) for k in ('rewrite_latency_ms','retrieval_latency_ms','answer_latency_ms')]
                row['total_latency_ms']=round(sum(float(x) for x in times if x is not None),2) if any(x is not None for x in times) else None
                final.append(row)
        print(f'[done] {ds}: {len(cases)*len(conditions)} conditions evaluated',flush=True)
        (out/'records.jsonl').write_text('\n'.join(json.dumps(x,ensure_ascii=False) for x in final)+'\n',encoding='utf-8')
    summary_and_stats(final,out,args.model,generic_fingerprint)
    from report import generate
    print('HTML report:', generate(out).resolve())


def parser():
    p=argparse.ArgumentParser(description='Controlled Zero / same-data / cross-data / generic 3-shot CQR study.')
    p.add_argument('--model',default='gpt-6-luna')
    p.add_argument('--expected-n',type=int,default=100)
    p.add_argument('--data-dir',default='data')
    p.add_argument('--generic-demos',default=str(Path(__file__).resolve().with_name('generic_demos.json')),
                   help='JSON list of exactly 3 fixed dataset-independent examples')
    p.add_argument('--out-dir',default='results')
    p.add_argument('--workers',type=int,default=5)
    p.add_argument('--qrecc-mode',choices=['proxy','external','rewrite_only'],default='proxy')
    p.add_argument('--qrecc-docs',help='For external mode only: JSONL of {"id": original passage id, "text": passage text}; include positives AND distractors.')
    p.add_argument('--dense-model',default='sentence-transformers/all-MiniLM-L6-v2')
    p.add_argument('--cross-encoder',default='cross-encoder/ms-marco-MiniLM-L-6-v2')
    p.add_argument('--candidates',type=int,default=30)
    p.add_argument('--no-raw',action='store_true',help='Omit no-rewrite additional baseline.')
    p.add_argument('--skip-retrieval',action='store_true')
    p.add_argument('--skip-answer',action='store_true')
    p.add_argument('--abort-on-error',action='store_true')
    return p

if __name__=='__main__':
    run(parser().parse_args())
