class BaseEmbeddings:
    def embed_texts(self, texts):
        raise NotImplementedError
    def name(self):
        return 'unknown'
