from pathlib import Path
import fitz
from PIL import Image
root=Path.cwd();assert (root/'.git').exists()
(root/'notes/rendered').mkdir(exist_ok=True)
for p in sorted((root/'papers/pages').glob('*_chunk_*.pdf')):
 doc=fitz.open(p); ims=[]
 for page in doc:
  pix=page.get_pixmap(matrix=fitz.Matrix(1.3,1.3));ims.append(Image.frombytes('RGB',[pix.width,pix.height],pix.samples))
 im=Image.new('RGB',(sum(x.width for x in ims),max(x.height for x in ims)),'white');x=0
 for a in ims:im.paste(a,(x,0));x+=a.width
 im.save(root/'notes/rendered'/f'{p.stem}.png')
