"""Exploratory update-matched sham, added after noticing format-only changes."""
import json
from pathlib import Path
from experiment import Harness,dump
from data import suite

def main():
    h=Harness();rows=suite('add5');base_lp=h.logps([r['prompt'] for r in rows]);info=h.train('add5','true20_kl10',11,steps=20,force_step=20);info.update(h.evaluate(rows,info['run'],base_lp))
    manifest=json.loads(Path('results/run_manifest.json').read_text());manifest=[r for r in manifest if r['run']!=info['run']]+[info];dump('results/run_manifest.json',manifest)
if __name__=='__main__':main()
