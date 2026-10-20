import pytest
import json
from pathlib import Path
from core import (build_corpus, choose_demos, rewrite_prompt, norm, score_answer, subspan_em,
                  retrieval_scores, bootstrap_diff, select_cases, validate_generic_demos, MODES)
from prepare import qrecc_case, canard_case


def row(i, group=None):
    return {'id':f'x:{i}','group':group or f'g:{i}','history':['Alice moved to Rome.'],
            'question':f'When did she move {i}?','gold_rewrite':f'When did Alice move to Rome {i}?',
            'answer':f'199{i}','evidence':f'In 199{i} Alice moved to Rome.', 'distractors':['This article discusses cars.'*10]}


def test_three_shot_no_leak():
    train=[row(i) for i in range(12)]
    evaluation=[row(100),row(101)]
    demos=choose_demos(train,3,42,evaluation)
    assert len(demos)==3
    assert set(c['group'] for c in demos).isdisjoint(c['group'] for c in evaluation)
    assert len([x for x in rewrite_prompt(evaluation[0],demos).split('Example ') if x.startswith(('1:','2:','3:'))])==3
    assert 'Example' not in rewrite_prompt(evaluation[0],[])


def test_retrieval_metrics():
    s=retrieval_scores(['noise','gold','more'],{'gold'})
    assert s['mrr_at_10']==0.5
    assert s['recall_at_5']==1.0
    assert s['ndcg_at_10'] < 1.0


def test_case_parsing():
    q=qrecc_case({'Conversation_source':'nq','Question':'What about him?',
        'Truth_rewrite':'What about Einstein?', 'Truth_answer':'scientist',
        'Context':['Who is Einstein?','A scientist'], 'Turn_no':2,
        'Conversation_no':10,'Truth_passages':['12']})
    assert q['dataset']=='qrecc'
    assert qrecc_case({**{'Conversation_source':'quac'},**{'Question':'x','Truth_rewrite':'x','Truth_answer':'y','Context':['a'],'Turn_no':2,'Conversation_no':1}}) is None
    c=canard_case({'Question':'When?','Rewrite':'When did she move?',
        'answer':'in 1920','true_contexts':'She moved in 1920 to Paris.',
        'QuAC_dialog_id':'C1','Question_no':2,'History':['Alice?'],'retrieved_contexts':['wrong']})
    assert c['dataset']=='canard'


def test_proxy_explicit_label():
    docs,qrels,label=build_corpus('qrecc',[{**row(1),'gold_passages':['1']}], 'proxy')
    assert 'NOT_OFFICIAL' in label
    assert qrels['x:1']
    with pytest.raises(ValueError):
        build_corpus('qrecc',[{**row(1),'gold_passages':['1']}], 'external',external_docs=[{'id':'2','text':'non-gold'}])


def test_bootstrap_matched():
    a=[{'case_id':'a','f1':1.},{'case_id':'b','f1':1.}]
    b=[{'case_id':'a','f1':0.},{'case_id':'b','f1':0.}]
    r=bootstrap_diff(a,b,'f1')
    assert r['n']==2 and r['delta']==1 and r['ci95']==[1.,1.]


def test_answer_em():
    assert score_answer('The Blue Fox','blue fox')['em']==1.0
    assert score_answer('blue','blue fox')['f1']>0


def test_answer_subspan_em():
    assert subspan_em('The answer is Blue Fox, a small mammal.', 'blue fox') == 1.0
    assert subspan_em('blue mammal', 'blue fox') == 0.0
    assert subspan_em('anything', '  ') == 0.0


def test_generic_examples_hand_authored_and_nonoverlapping():
    demos=json.loads((Path(__file__).resolve().parents[1]/'generic_demos.json').read_text())
    assert MODES == ('zero_shot', 'same_3shot', 'cross_3shot', 'generic_3shot')
    validate_generic_demos(demos,[row(100),row(101)])
    prompt=rewrite_prompt(row(100),demos)
    assert prompt.count('Standalone rewrite:')==4  # 3 examples + query being evaluated
    assert [x['type'] for x in demos] == ['coreference_resolution','ellipsis_and_parallel_question','topic_switching']
    with pytest.raises(ValueError, match='exactly 3'):
        validate_generic_demos(demos[:2])
    with pytest.raises(ValueError, match='matches evaluation'):
        validate_generic_demos(demos,[{'question':demos[0]['question'],'gold_rewrite':'unrelated'}])
