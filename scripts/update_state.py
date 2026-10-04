from pathlib import Path
import sys
r=Path(__file__).resolve().parents[1]
assert Path.cwd()==r and (r/'.git').exists()
p=r/'STATE.md';s=p.read_text();a='<!-- NEURICO_AGENT_NOTES_START:resource_finder -->';b='<!-- NEURICO_AGENT_NOTES_END:resource_finder -->'
p.write_text(s.split(a)[0]+a+'\n'+sys.stdin.read().strip()+'\n'+b+s.split(b,1)[1])
