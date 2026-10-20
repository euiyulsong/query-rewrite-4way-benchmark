"""Recompute paired statistics without new OpenAI API calls."""
import argparse
import json
from pathlib import Path
from run import read_jsonl, summary_and_stats

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--out-dir',default='results')
    p.add_argument('--model',default='gpt-6-luna')
    a=p.parse_args()
    summary_and_stats(read_jsonl(Path(a.out_dir)/'records.jsonl'),a.out_dir,a.model)
