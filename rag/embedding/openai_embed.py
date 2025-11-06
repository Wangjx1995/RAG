from openai import OpenAI
import os, numpy as np
from .base import BaseEmbeddings
class OpenAIEmbedding(BaseEmbeddings):
    def __init__(self, model=None, api_key=None, base_url=None):
        self.model = model or os.getenv('OPENAI_EMBEDDING_MODEL','text-embedding-3-small')
        self.client = OpenAI(api_key=api_key or os.getenv('OPENAI_API_KEY'), base_url=base_url or os.getenv('OPENAI_BASE_URL') or None)
    def embed_texts(self, texts):
        r = self.client.embeddings.create(model=self.model, input=texts)
        return np.array([d.embedding for d in r.data], dtype='float32')
    def name(self):
        return f'openai:{self.model}'
