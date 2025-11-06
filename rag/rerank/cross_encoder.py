from sentence_transformers import CrossEncoder
class CrossEncoderReranker:
    def __init__(self, model_name='cross-encoder/ms-marco-MiniLM-L-6-v2'):
        self.model=CrossEncoder(model_name)
    def rerank(self, query, docs, top_k=5):
        if not docs: return []
        pairs=[(query, d.get('text','')) for d in docs]
        scores=self.model.predict(pairs).tolist()
        for d,s in zip(docs, scores): d['re_rank']=float(s)
        docs=sorted(docs, key=lambda x:x.get('re_rank',0.0), reverse=True)
        return docs[:top_k]
