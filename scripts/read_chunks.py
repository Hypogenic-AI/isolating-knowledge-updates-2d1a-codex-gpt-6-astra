from pathlib import Path
from pypdf import PdfReader
root=Path.cwd(); assert (root/'.git').exists()
for p in sorted((root/'papers/pages').glob('*_chunk_*.pdf')):
 text='\n'.join(page.extract_text() for page in PdfReader(p).pages)
 (root/'notes'/f'{p.stem}.txt').write_text(text)
print('Chunks extracted',len(list((root/'notes').glob('*_chunk_*.txt'))))
