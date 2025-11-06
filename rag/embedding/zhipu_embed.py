import os, requests, numpy as np
from .base import BaseEmbeddings
class ZhipuEmbedding(BaseEmbeddings):
    def __init__(self, model=None, api_key=None, base_url=None):
        self.model = model or os.getenv('ZHIPU_EMBEDDING_MODEL','text-embedding-3')
        self.api_key = api_key or os.getenv('ZHIPU_API_KEY')
        self.base_url = (base_url or os.getenv('ZHIPU_BASE_URL') or 'https://open.bigmodel.cn/api').rstrip('/')
        if not self.api_key:
            raise RuntimeError('ZHIPU_API_KEY 未配置')
    def embed_texts(self, texts):
        url=f"{self.base_url}/embeddings"; headers={'Authorization':f'Bearer {self.api_key}'}; payload={'model':self.model,'input':texts}
        r=requests.post(url, headers=headers, json=payload, timeout=60); r.raise_for_status(); data=r.json()
        return np.array([d['embedding'] for d in data['data']], dtype='float32')
    def name(self):
        return f'zhipu:{self.model}'
