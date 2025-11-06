import os, requests, numpy as np
from .base import BaseEmbeddings
class JinaEmbedding(BaseEmbeddings):
    def __init__(self, model=None, api_key=None, base_url=None):
        self.model = model or os.getenv('JINA_EMBEDDING_MODEL','jina-embeddings-v2-base-ja')
        self.api_key = api_key or os.getenv('JINA_API_KEY')
        self.base_url = (base_url or os.getenv('JINA_BASE_URL') or 'https://api.jina.ai/v1').rstrip('/')
        if not self.api_key:
            raise RuntimeError('JINA_API_KEY 未配置')
    def embed_texts(self, texts):
        url=f"{self.base_url}/embeddings"; headers={'Authorization':f'Bearer {self.api_key}'}; payload={'model':self.model,'input':texts}
        r=requests.post(url, headers=headers, json=payload, timeout=60); r.raise_for_status(); data=r.json()
        return np.array([d['embedding'] for d in data['data']], dtype='float32')
    def name(self):
        return f'jina:{self.model}'
