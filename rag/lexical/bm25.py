from rank_bm25 import BM25Okapi
import re, os, json
class BM25Index:
    def __init__(self, docs):
        self.docs=docs; self.corpus=[self._tok(d.get('text','')) for d in docs]; self.bm25=BM25Okapi(self.corpus)
    def _tok(self,t):
        return [w for w in re.split(r'\W+', t.lower()) if w]
    def query(self,q,k=30):
        scores=self.bm25.get_scores(self._tok(q)); idx=sorted(range(len(scores)), key=lambda i:-scores[i])[:k]
        out=[]
        for i in idx:
            d=dict(self.docs[i]); d['bm25']=float(scores[i]); out.append(d)
        return out
    def persist(self,path='storage'):
        os.makedirs(path,exist_ok=True)
        slim=[{'id':d.get('id'),'text':d.get('text',''),'source':d.get('source'),'chunk_id':d.get('chunk_id')} for d in self.docs]
        with open(os.path.join(path,'bm25.json'),'w',encoding='utf-8') as f:
            json.dump({'docs':slim},f,ensure_ascii=False)
    @staticmethod
    def load(path='storage'):
        p=os.path.join(path,'bm25.json')
        with open(p,'r',encoding='utf-8') as f:
            return BM25Index(json.load(f)['docs'])
