import os, json, numpy as np
class VectorStore:
    def __init__(self, docs=None, emb=None, embedding_model_name='openai:text-embedding-3-small'):
        self.docs=docs or []; self.emb=emb; self.embedding_model_name=embedding_model_name
    def get_vector(self, embedding):
        import numpy as np
        X=embedding.embed_texts([d['text'] for d in self.docs]).astype('float32')
        X=X/(np.linalg.norm(X,axis=1,keepdims=True)+1e-12)
        self.emb=X; self.embedding_model_name=embedding.name(); return self
    def persist(self,path='storage'):
        os.makedirs(path,exist_ok=True)
        np.save(os.path.join(path,'emb.npy'), self.emb)
        with open(os.path.join(path,'index.json'),'w',encoding='utf-8') as f:
            json.dump({'embedding_model':self.embedding_model_name,'docs':self.docs,'count':len(self.docs),'dim':int(self.emb.shape[1])}, f, ensure_ascii=False)
    def load_vector(self,path='storage'):
        with open(os.path.join(path,'index.json'),'r',encoding='utf-8') as f:
            meta=json.load(f)
        self.docs=meta['docs']; self.embedding_model_name=meta.get('embedding_model',self.embedding_model_name)
        self.emb=np.load(os.path.join(path,'emb.npy'))
        return self
    def query(self, qtext, embedding, k=4):
        if self.emb is None or not self.docs: return []
        import numpy as np
        q=embedding.embed_texts([qtext])[0].astype('float32'); q=q/(np.linalg.norm(q)+1e-12)
        sims=self.emb@q; idx=np.argsort(-sims)[:k]; out=[]
        for i in idx:
            d=dict(self.docs[i]); d['score']=float(sims[i]); out.append(d)
        return out
