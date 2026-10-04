"""Re-run deterministic CPU analysis and compare hashes of all key numeric artifacts."""
import hashlib,json,subprocess,sys
from pathlib import Path

def main():
    paths=['summary.csv','scored.csv','stratum_metrics.csv','descriptive.csv','cluster_intervals.json','paired_tests.json','arithmetic_diagnostics.csv','zero_operand_diagnostics.csv','api_summary.csv','api_cost.json','parser_sensitivity.csv','selected_budgets.csv']
    def hashes():return {p:hashlib.sha256((Path('results')/p).read_bytes()).hexdigest() for p in paths}
    before=hashes()
    with Path('logs/analysis_repeat.log').open('w') as f:
        for script in ['analyze.py','sensitivity.py','budgets.py']:subprocess.run([sys.executable,'src/'+script],stdout=f,stderr=subprocess.STDOUT,check=True)
    after=hashes();result={'files_checked':len(paths),'all_identical':before==after,'before':before,'after':after}
    Path('results/analysis_reproduction.json').write_text(json.dumps(result,indent=2));assert before==after,'Analysis artifacts changed';print('All numeric analysis artifacts reproduced identically')
if __name__=='__main__':main()
