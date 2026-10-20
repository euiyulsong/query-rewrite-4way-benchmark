"""Write a local, dependency-free HTML result browser from completed records."""
from __future__ import annotations
import html
import json
from pathlib import Path


def esc(x): return html.escape(str(x if x is not None else '—'))


def fmt(x, percent=False):
    if x is None: return '—'
    if isinstance(x,(int,float)):
        return f'{float(x)*100:.1f}%' if percent else f'{float(x):.3f}'
    return str(x)


def generate(out_dir):
    out=Path(out_dir)
    summary=json.loads((out/'summary.json').read_text(encoding='utf-8'))
    records=[json.loads(line) for line in (out/'records.jsonl').read_text(encoding='utf-8').splitlines() if line]
    parts=['''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Query Rewrite · 4-way Benchmark</title><style>
    :root{font-family:ui-sans-serif,system-ui,Arial,sans-serif;color:#182331;background:#f6f8fb}
    body{max-width:1150px;margin:35px auto;padding:0 18px}h1{letter-spacing:-1px;margin-bottom:3px}h2{margin-top:32px}
    p{line-height:1.65}small,.muted{color:#66758b}.banner{background:#fff7e8;border:1px solid #f0cea2;padding:14px;border-radius:12px;font-size:14px}
    .panel{border:1px solid #e6ebf1;border-radius:14px;padding:20px;background:#fff;margin:14px 0;overflow:auto}
    table{width:100%;border-collapse:collapse;font-size:13px}td,th{padding:12px;text-align:left;border-bottom:1px solid #e9edf2;vertical-align:top}
    th{background:#f6f8fc;white-space:nowrap}.metric{font-variant-numeric:tabular-nums;text-align:right;white-space:nowrap}
    .pill{background:#e9f1ff;color:#1b5086;padding:4px 9px;border-radius:12px;font-size:12px;white-space:nowrap}
    .flex{display:flex;gap:10px;flex-wrap:wrap;align-items:center}.flex label{font-weight:600}
    select,input{border:1px solid #dbe3ed;padding:9px;border-radius:8px;background:white;font-size:14px}
    details{border-bottom:1px solid #ebeff5;padding:12px 3px}details summary{cursor:pointer;font-weight:600}
    .c{padding:10px;margin-top:8px;background:#f6f8fc;border-radius:8px;font-size:13px;overflow-wrap:anywhere;line-height:1.6}
    .loss{color:#af3c34}.gain{color:#17754c}code{font-size:13px}
    </style></head><body>''']
    parts.append('<h1>Query Rewrite · GPT-6 Luna</h1><p class="muted">Zero-shot vs same-data 3-shot vs cross-data 3-shot vs generic 3-shot · 100 per dataset</p>')
    for w in summary.get('warnings',[]):
        parts.append(f'<div class="banner">{esc(w)}</div>')
    ordered=['raw_question','zero_shot','same_3shot','cross_3shot','generic_3shot']
    for ds in ('qrecc','canard'):
        parts.append(f'<h2>{ds.upper()}</h2><div class="panel"><table><thead><tr><th>Condition</th><th>Rewrite F1</th><th>Recall@5</th><th>MRR@10</th><th>nDCG@10</th><th>Answer EM</th><th>Answer F1</th><th>Rewrite ms</th><th>Input tokens</th><th>Total ms</th></tr></thead><tbody>')
        for mode in ordered:
            r=summary['groups'].get(f'{ds}/{mode}')
            if not r: continue
            parts.append('<tr><td><span class="pill">'+esc(mode)+'</span></td>'+''.join(
                f'<td class="metric">{esc(fmt(r.get(key),pct))}</td>' for key,pct in (
                ('rewrite_f1',True),('recall_at_5',True),('mrr_at_10',True),
                ('ndcg_at_10',True),('answer_em',True),('answer_f1',True),
                ('rewrite_latency_ms',False),('rewrite_input_tokens',False),('total_latency_ms',False)))+'</tr>')
        parts.append('</tbody></table></div>')
        parts.append('<div class="panel"><h3>Paired differences (candidate minus zero-shot)</h3><table><tr><th>Condition</th><th>Metric</th><th>Delta</th><th>95% bootstrap CI</th><th>Wins / Losses / Ties</th></tr>')
        for mode in ('same_3shot','cross_3shot','generic_3shot'):
            d=summary['paired_differences'].get(f'{ds}/{mode}_minus_zero_shot',{})
            for metric in ('rewrite_f1','mrr_at_10','answer_em'):
                rec=d.get(metric,{})
                case=summary.get('case_level_comparison',{}).get(f'{ds}/{mode}/{metric}',{})
                delta=rec.get('delta')
                cls='gain' if delta is not None and delta>0 else 'loss' if delta is not None and delta<0 else ''
                parts.append(f'<tr><td>{esc(mode)}</td><td>{esc(metric)}</td><td class="metric {cls}">{esc(fmt(delta))}</td>'
                             f'<td>{esc(rec.get("ci95"))}</td><td>{case.get("wins",0)} / {case.get("losses",0)} / {case.get("ties",0)}</td></tr>')
        parts.append('</table></div>')

    parts.append('<h2>Paired per-query inspection</h2><div class="panel"><div class="flex"><label for="dataset">Dataset</label>'
                 '<select id="dataset" onchange="filterRows()"><option value="all">All</option><option value="qrecc">QReCC</option><option value="canard">CANARD</option></select>'
                 '<label for="search">Search</label><input id="search" oninput="filterRows()" placeholder="question / rewrite" />'
                 '<label for="compare">Few-shot type</label><select id="compare" onchange="filterRows()">'
                 '<option value="any">Any Few-shot</option><option value="same_3shot">Same-data</option><option value="cross_3shot">Cross-data</option><option value="generic_3shot">Generic</option></select>'
                 '<label for="status">Comparison</label><select id="status" onchange="filterRows()">'
                 '<option value="all">All</option><option value="loss">Few-shot loss</option><option value="gain">Few-shot gain</option><option value="tie">Tie</option></select></div>')
    groups={}
    for r in records:
        groups.setdefault((r['dataset'],r['case_id']),{})[r['condition']]=r
    for (ds,cid),data in sorted(groups.items()):
        base=data.get('zero_shot')
        if not base:continue
        comparisons=[x for x in ('same_3shot','cross_3shot','generic_3shot') if x in data]
        if not comparisons:continue
        mode_status={}
        for m in comparisons:
            a,b=data[m].get('mrr_at_10'),base.get('mrr_at_10')
            if a is None or b is None:
                a,b=data[m].get('rewrite_f1'),base.get('rewrite_f1')
            mode_status[m]=('unknown' if a is None or b is None else
                            'loss' if a<b else 'gain' if a>b else 'tie')
        statuses=list(mode_status.values())
        status='loss' if 'loss' in statuses else 'gain' if 'gain' in statuses else 'tie' if 'tie' in statuses else 'unknown'
        data_search=' '.join([base.get('question','')]+[r.get('rewrite','') for r in data.values()])
        parts.append(f'<details class="example" data-dataset="{esc(ds)}" data-status="{esc(status)}" data-comparisons="{esc(json.dumps(mode_status))}" data-search="{esc(data_search.lower())}">'
                     f'<summary>{esc(ds.upper())} · {esc(cid)} · {esc(base.get("question",""))[:120]} · '
                     f'<span class="{status if status in ("loss","gain") else "muted"}">{esc(status)}</span></summary>')
        parts.append(f'<div class="c"><b>Gold rewrite:</b> {esc(base.get("gold_rewrite"))}<br><b>Gold answer:</b> {esc(base.get("gold_answer"))}</div>')
        for mode in ordered:
            r=data.get(mode)
            if not r:continue
            parts.append(f'<div class="c"><b>{esc(mode)}</b> · MRR@10 {esc(fmt(r.get("mrr_at_10")))} · Rewrite F1 {esc(fmt(r.get("rewrite_f1")))} · Answer EM {esc(fmt(r.get("answer_em")))}<div>{esc(r.get("rewrite"))}</div>'
                         + (f'<div><b>Predicted answer:</b> {esc(r.get("predicted_answer"))}</div>' if r.get('predicted_answer') else '')+'</div>')
        parts.append('</details>')
    parts.append('''</div><script>function filterRows(){const ds=document.getElementById('dataset').value,s=document.getElementById('search').value.toLowerCase(),st=document.getElementById('status').value,cmp=document.getElementById('compare').value;
      for(const el of document.querySelectorAll('details.example')){el.style.display=((ds==='all'||el.dataset.dataset===ds)&&(st==='all'||(cmp==='any'?el.dataset.status:(JSON.parse(el.dataset.comparisons)[cmp]||'unknown'))===st)&&el.dataset.search.includes(s))?'':'none';}}</script></body></html>''')
    (out/'report.html').write_text(''.join(parts),encoding='utf-8')
    return out/'report.html'

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--out-dir',default='results')
    args=p.parse_args(); print(generate(args.out_dir))
