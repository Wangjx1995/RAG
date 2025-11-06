import os
from pypdf import PdfReader

def _iter_files(root, exts):
    exts=tuple(e.lower() for e in exts)
    for dp,_,fs in os.walk(root):
        for n in fs:
            if n.lower().endswith(exts):
                yield os.path.join(dp,n)

def read_text_file(p):
    return open(p,'r',encoding='utf-8',errors='ignore').read()

def read_pdf(p):
    r=PdfReader(p); txt=[]
    for pg in r.pages:
        try: txt.append(pg.extract_text() or '')
        except: pass
    return '\n'.join(txt)

def chunk_text(t, chunk_chars=600, overlap=150):
    ch=[]; i=0; n=len(t)
    while i<n:
        e=min(n,i+chunk_chars); ch.append(t[i:e]); i=(e-overlap) if (e-overlap)>i else e
    return ch

class ReadFiles:
    def __init__(self, root, exts=("txt","md","html","htm","pdf")):
        self.root=root; self.exts=exts
    def get_content(self, chunk_chars=600, overlap=150):
        docs=[]
        for p in _iter_files(self.root, self.exts):
            raw=read_pdf(p) if p.lower().endswith('.pdf') else read_text_file(p)
            for i,ch in enumerate(chunk_text(raw, chunk_chars, overlap)):
                docs.append({'id':f"{os.path.relpath(p,self.root)}#chunk{i}", 'text':ch, 'source':os.path.relpath(p,self.root), 'chunk_id':i})
        return docs
