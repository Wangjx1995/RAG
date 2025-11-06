import os, re
from pypdf import PdfReader

def _iter_files(root, exts):
    exts=tuple(e.lower() for e in exts)
    for dp,_,fs in os.walk(root):
        for n in fs:
            if n.lower().endswith(exts):
                yield os.path.join(dp,n)

def _read_text_file(p):
    return open(p,'r',encoding='utf-8',errors='ignore').read()

def _read_pdf(p):
    r=PdfReader(p); txt=[]
    for pg in r.pages:
        try: txt.append(pg.extract_text() or '')
        except: pass
    return '\n'.join(txt)

def chunk_by_sentences(text, target_chars=700, overlap=150):
    blocks=re.split(r'(?<=[。！？!?．\.])\s*', text)
    chunks=[]; cur=''
    for s in blocks:
        if not s: continue
        if len(cur)+len(s)<=target_chars:
            cur+=s
        else:
            if cur: chunks.append(cur); cur=cur[max(0,len(cur)-overlap):]+s
            else: chunks.append(s)
    if cur: chunks.append(cur)
    return chunks

class ReadFilesSent:
    def __init__(self, root, exts=("txt","md","html","htm","pdf")):
        self.root=root; self.exts=exts
    def get_content(self, target_chars=700, overlap=150):
        docs=[]
        for p in _iter_files(self.root, self.exts):
            raw=_read_pdf(p) if p.lower().endswith('.pdf') else _read_text_file(p)
            for i,ch in enumerate(chunk_by_sentences(raw, target_chars, overlap)):
                docs.append({'id':f"{os.path.relpath(p,self.root)}#chunk{i}", 'text':ch, 'source':os.path.relpath(p,self.root), 'chunk_id':i})
        return docs
