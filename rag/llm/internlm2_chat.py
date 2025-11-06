from openai import OpenAI
import os
class InternLM2Chat:
    def __init__(self, model=None):
        self.model = model or os.getenv('INTERNLM2_CHAT_MODEL','internlm2-chat')
        base=os.getenv('INTERNLM2_BASE_URL'); key=os.getenv('INTERNLM2_API_KEY')
        if not (base and key):
            raise RuntimeError('INTERNLM2_BASE_URL / INTERNLM2_API_KEY 未配置')
        self.client = OpenAI(api_key=key, base_url=base)
    def chat(self, question, contexts, max_ref=5):
        ctx='\n\n---\n\n'.join([f"[{c.get('source')}#chunk{c.get('chunk_id')}]\n{c.get('text')}" for c in contexts[:max_ref]]) or '(コンテキストなし)'
        sys='与えられた資料のみで回答し、最後に出典を列挙してください。'
        user=f'質問：{question}\n\n資料：\n{ctx}'
        r=self.client.chat.completions.create(model=self.model,messages=[{'role':'system','content':sys},{'role':'user','content':user}],temperature=0.2)
        return r.choices[0].message.content.strip()
