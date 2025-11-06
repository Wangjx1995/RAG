import argparse, os
from dotenv import load_dotenv,find_dotenv
from rag.utils import ReadFiles
from rag.utils_extras import ReadFilesSent
from rag.embedding.openai_embed import OpenAIEmbedding
from rag.embedding.jina_embed import JinaEmbedding
from rag.embedding.zhipu_embed import ZhipuEmbedding
from rag.vector_store import VectorStore
from rag.lexical.bm25 import BM25Index

def make_embedding(backend, model):
    b=(backend or 'openai').lower()
    return OpenAIEmbedding(model) if b=='openai' else JinaEmbedding(model) if b=='jina' else ZhipuEmbedding(model) if b=='zhipu' else (_ for _ in ()).throw(ValueError('Unknown embed backend'))

def main():
    ap=argparse.ArgumentParser();
    ap.add_argument('--data',default='data'); ap.add_argument('--storage',default='storage')
    ap.add_argument('--chunk',type=int,default=600); ap.add_argument('--overlap',type=int,default=150)
    ap.add_argument('--chunker',choices=['char','sent'],default='char')
    ap.add_argument('--embed-backend',choices=['openai','jina','zhipu'],default='openai')
    ap.add_argument('--embed-model',default=None)
    args=ap.parse_args(); load_dotenv(find_dotenv(filename=".env", usecwd=True), override=True)
    docs=ReadFiles(args.data).get_content(args.chunk,args.overlap) if args.chunker=='char' else ReadFilesSent(args.data).get_content(args.chunk,args.overlap)
    emb=make_embedding(args.embed_backend, args.embed_model); store=VectorStore(docs).get_vector(emb)
    os.makedirs(args.storage,exist_ok=True); store.persist(args.storage)
    try:
        BM25Index(docs).persist(args.storage); print('[INFO] BM25 Index stored')
    except Exception as e:
        print('[WARN] BM25 Persistence failed:', e)
if __name__=='__main__': main()
