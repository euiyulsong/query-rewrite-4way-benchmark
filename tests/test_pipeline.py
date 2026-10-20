import json
from pathlib import Path
import run as runner


def test_local_end_to_end_fake_api_and_retriever(tmp_path, monkeypatch):
    class FakeGPT:
        def __init__(self,model,cache_dir): pass
        def ask(self,task,system,prompt,max_tokens):
            return {'text':'When did Alice visit Paris?' if task=='rewrite' else '2001',
                    'latency_ms':14, 'input_tokens':40,'output_tokens':7}
    class FakeRetriever:
        def __init__(self,docs,*args): self.docs=docs
        def search(self,query,k=10):return self.docs[:k], 5
    monkeypatch.setattr(runner,'GPTClient',FakeGPT)
    monkeypatch.setattr(runner,'HybridRetriever',FakeRetriever)
    examples=[]
    for i in range(2):
        examples.append({'id':f'test:{i}','group':f'group:{i}',
                         'history':['Alice visited Paris.'],'question':'When did she visit?',
                         'gold_rewrite':'When did Alice visit Paris?', 'answer':'2001',
                         'evidence':'In 2001, Alice visited Paris on a work trip.',
                         'distractors':['Cars were made in a factory for several months.'],
                         'gold_passages':['1']})
    dd=tmp_path/'data';dd.mkdir()
    for ds in ('qrecc','canard'):
        (dd/f'{ds}.json').write_text(json.dumps({'eval':[{**x,'id':f'{ds}:{i}'} for i,x in enumerate(examples)],
                                                'demos':[{**x,'id':f'{ds}:demo:{i}'} for i,x in enumerate(examples*2)][:3]}))
    args=runner.parser().parse_args(['--data-dir',str(dd),'--out-dir',str(tmp_path/'results'),
                                      '--expected-n','2','--workers','2'])
    runner.run(args)
    out=tmp_path/'results'
    assert (out/'report.html').exists()
    assert (out/'summary.csv').exists()
    result=json.loads((out/'summary.json').read_text())
    assert result['groups']['qrecc/zero_shot']['evaluated_cases']==2
    assert result['groups']['canard/same_3shot']['answers_completed']==2
    assert len((out/'records.jsonl').read_text().splitlines())==20
    assert result['groups']['qrecc/generic_3shot']['evaluated_cases']==2
    assert result['groups']['canard/generic_3shot']['answers_completed']==2
    assert 'qrecc/generic_3shot_minus_zero_shot' in result['paired_differences']
    assert 'canard/generic_3shot/mrr_at_10' in result['case_level_comparison']
    assert result['generic_demo_fingerprint']
    html=(out/'report.html').read_text()
    assert 'generic_3shot' in html
    assert 'Few-shot type' in html
    assert 'data-comparisons=' in html
    assert '&quot;generic_3shot&quot;' in html
