from pathlib import Path
import json,hashlib,re,csv
from pypdf import PdfReader
R=Path(__file__).resolve().parents[1];assert Path.cwd()==R and (R/'.git').exists()
core=json.loads((R/'notes/paper_metadata.json').read_text()); search=[json.loads(l) for l in (R/'notes/search_downloads_retry.jsonl').read_text().splitlines()]
entries={}
for x in search:
 if x.get('file') and (R/x['file']).exists(): entries[Path(x['file']).name]=x
for x in core: entries[x['id']+'.pdf']={**entries.get(x['id']+'.pdf',{}),**x}
rows=[]
for p in sorted((R/'papers').glob('*.pdf')):
 x=entries.get(p.name,{});t=PdfReader(p);title=x.get('title',t.metadata.title if t.metadata else p.stem)
 authors=x.get('authors','not resolved');authors='; '.join(authors) if isinstance(authors,list) else authors
 status='Abstract/title screening only'
 if p.stem in ['2307.12976','2407.12828','2401.07453','2402.11078','2401.01286','2410.21272','2502.00873','2409.01659','2410.17194']:status='All PDF chunks visually reviewed; notes/chunk_reading_log.md'
 if p.stem=='2510.17941':status='Main text pp1–12 reviewed; linked blog fully read; appendices archived only'
 rows.append(dict(title=title,authors=authors,year=x.get('year','20'+p.stem[:2] if re.match(r'\d{4}',p.stem) else 'unknown'),source=x.get('pdf_url',x.get('url','')),file=str(p.relative_to(R)),pages=len(t.pages),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),reading=status))
(R/'notes/papers_manifest.json').write_text(json.dumps(rows,indent=2))
lines=['# Downloaded papers','',f'{len(rows)} original PDFs; page chunks are derivative reading copies, not additional papers. All PDFs parse successfully. Full metadata/checksums: `notes/papers_manifest.json`. User-specified arXiv sources were prioritized. Three user-specified web articles are archived in `notes/` (see below).','', '| Paper (first submission year) | Authors | Pages | Reading status |','|---|---|---:|---|']
for x in rows:lines.append(f"| [{x['title']}]({Path(x['file']).name}) ({x['year']}) | {x['authors']} | {x['pages']} | {x['reading']} |")
lines+=['','## Specified web articles','', '- [Modifying beliefs via SDF](https://alignment.anthropic.com/2025/modifying-beliefs-via-sdf/): local `notes/sdf.html`, `notes/sdf.txt`; full article/appendix reviewed, including correction.', '- [Believe It or Not](https://alignment.anthropic.com/2025/believe-it-or-not/): local `notes/believe.html`, `notes/believe.txt`; fully reviewed; linked paper 2510.17941 also downloaded.', '- [What happens when you train models on false facts?](https://www.lesswrong.com/posts/CdymgH4MQdFgB6Fg7/what-happens-when-you-train-models-on-false-facts-1): local `notes/false_facts.html`, `notes/false_facts.txt`; full post reviewed, exploratory non-peer-reviewed evidence.','', '## Screening and access limitations','', 'Paper-finder returned 233 candidates, not 233 equally relevant papers. All entries had download attempts and failures had an alternate arXiv title lookup. Broad/off-domain hits are retained as archives, not added experiment directions. Semantic Scholar rate limits and arXiv/API errors blocked many supplementary downloads. See `notes/search_downloads_retry.jsonl` for per-entry errors and `notes/search_access.csv` for reconciled local availability. Citation counts are search metadata, not independently verified. First submission years may differ from revised PDF/conference years. Missing abstracts were screened by title; no full-text review is claimed for those entries.','', 'Reproduce: `uv sync`; `uv run python scripts/gather_papers.py`; `uv run python scripts/download_search_results.py`; `uv run python scripts/retry_search_downloads.py`. PDFs use the downloaded versions/checksums above; scripts using unversioned URLs may retrieve later revisions. All original and chunk PDFs stay in this directory; extracted notes/renderings stay under `notes/`.']
(R/'papers/README.md').write_text('\n'.join(lines)+'\n')
norm=lambda s:re.sub(r'[^a-z0-9]','',s.lower())
bytitle={norm(x['title']):x['file'] for x in rows}
with (R/'notes/search_access.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=['index','title','relevance','local_file','status','error']);w.writeheader()
 for x in search:
  p=bytitle.get(norm(x['title']),x.get('file',''));ok=bool(p and (R/p).exists())
  w.writerow(dict(index=x['index'],title=x['title'],relevance=x['relevance'],local_file=p if ok else '',status='available' if ok else 'unavailable_after_retry',error='' if ok else x.get('fallback_error',x.get('error','No exact match'))))
print('Cataloged',len(rows),'PDFs')
