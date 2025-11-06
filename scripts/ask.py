import argparse, os
from dotenv import load_dotenv, find_dotenv
from rag.embedding.openai_embed import OpenAIEmbedding
from rag.embedding.jina_embed import JinaEmbedding
from rag.embedding.zhipu_embed import ZhipuEmbedding
from rag.vector_store import VectorStore
from rag.llm.openai_chat import OpenAIChat
from rag.llm.internlm2_chat import InternLM2Chat

def make_embedding_by_name(name_str: str):
    if not name_str or ":" not in name_str:
        return OpenAIEmbedding()
    b, m = name_str.split(":", 1)
    b = b.lower()
    return OpenAIEmbedding(m) if b=="openai" else JinaEmbedding(m) if b=="jina" else ZhipuEmbedding(m) if b=="zhipu" else OpenAIEmbedding()

def make_embedding_override(backend: str|None, model: str|None):
    if not backend: return None
    b = backend.lower()
    return OpenAIEmbedding(model) if b=="openai" else JinaEmbedding(model) if b=="jina" else ZhipuEmbedding(model) if b=="zhipu" else (_ for _ in ()).throw(ValueError("Unknown embed backend override"))

def _normalize_inplace(items, key):
    xs = [float(d.get(key,0.0)) for d in items if key in d]
    if not xs:
        for d in items: d[key+"_norm"]=0.0
        return
    mn, mx = min(xs), max(xs)
    for d in items:
        v=float(d.get(key,0.0)); d[key+"_norm"]=0.0 if mx==mn else (v-mn)/(mx-mn)

def _hybrid_merge(vec_hits, lex_hits, w_vec=0.6, w_bm25=0.4, top_m=20):
    pool={}
    for d in vec_hits: pool[d["id"]]=dict(d)
    for d in lex_hits: pool.setdefault(d["id"],dict(d)); pool[d["id"]]["bm25"]=d.get("bm25",0.0)
    merged=list(pool.values()); _normalize_inplace(merged,"score"); _normalize_inplace(merged,"bm25")
    for d in merged: d["hybrid"]=w_vec*d.get("score_norm",0.0)+w_bm25*d.get("bm25_norm",0.0)
    merged.sort(key=lambda x: x.get("hybrid", x.get("score",0.0)), reverse=True)
    return merged[:top_m]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--storage", default="storage")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--q","--question", dest="question", default=None)
    ap.add_argument("--embed-backend", choices=["openai","jina","zhipu"], default=None)
    ap.add_argument("--embed-model", default=None)
    ap.add_argument("--llm-backend", choices=["openai","internlm2","none"], default=None)
    ap.add_argument("--llm-model", default=None)
    ap.add_argument("--pool", type=int, default=30)
    ap.add_argument("--no-bm25", action="store_true")
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument("--vec-weight", type=float, default=0.6)
    ap.add_argument("--bm25-weight", type=float, default=0.4)
    args = ap.parse_args()

    load_dotenv(find_dotenv(filename=".env", usecwd=True), override=True)
    if not args.question:
        args.question = input("質問を入力してください：").strip()

    store = VectorStore().load_vector(args.storage)
    emb = make_embedding_override(args.embed-backend, args.embed_model) if hasattr(args, "embed-backend") else None
    if not emb:
        emb = make_embedding_by_name(store.embedding_model_name)

    vec_hits = store.query(args.question, emb, k=max(args.pool, args.k))

    lex_hits = []
    if not args.no_bm25:
        try:
            from rag.lexical.bm25 import BM25Index
            bm25 = BM25Index.load(args.storage)
            lex_hits = bm25.query(args.question, k=args.pool)
        except Exception as e:
            print("[WARN] BM25 load/search failed:", e)

    top_m = max(args.k*4, args.pool)
    merged = _hybrid_merge(vec_hits, lex_hits, args.vec_weight, args.bm25_weight, top_m) if lex_hits else vec_hits[:top_m]

    hits = merged
    if not args.no_rerank:
        try:
            from rag.rerank.cross_encoder import CrossEncoderReranker
            hits = CrossEncoderReranker().rerank(args.question, merged, top_k=args.k)
        except Exception as e:
            print("[WARN] Rerank failed / no sentence-transformers:", e); hits = merged[:args.k]
    else:
        hits = merged[:args.k]

    
    backend = args.llm_backend
    if backend is None:
        backend = "openai" if os.getenv("OPENAI_API_KEY") else ("internlm2" if (os.getenv("INTERNLM2_BASE_URL") and os.getenv("INTERNLM2_API_KEY")) else "none")

    if backend == "openai":
        chat = OpenAIChat(model=args.llm_model or None)
    elif backend == "internlm2":
        chat = InternLM2Chat(model=args.llm_model or None)
    else:
        from rag.llm.no_llm import NoLLM
        chat = NoLLM()

    print(chat.chat(args.question, hits, max_ref=args.k))

if __name__ == "__main__":
    main()