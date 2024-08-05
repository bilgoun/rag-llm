import os
import json
import torch
import faiss
import requests
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware as StarletteSessionMiddleware
from transformers import AutoTokenizer, AutoModel
from transformers.models.qwen import QWenModel, QWenConfig
from qwen.configuration_qwen import QWenConfig
from qwen


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model_name = "Qwen/Qwen-7B-Chat"
remote_model_url = "http://219.117.47.40:8000/v1/chat/completions"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
config = QWenConfig.from_pretrained(model_name, trust_remote_code=True)
embedder = AutoModel.from_pretrained(model_name, config=config trust_remote_code=True).to(device)

file_path = '/home/bilguun/ai-lab/research/output.txt'
with open(file_path, 'r', encoding='utf-8') as f:
    text = f.read().strip()

def chunk_document_japanese(text, chunk_size=512):
    input_ids = tokenizer.encode(text, truncation=False)
    chunks = []
    for i in range(0, len(input_ids), chunk_size):
        chunk_ids = input_ids[i:i + chunk_size]     
        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True)         
        chunks.append(chunk_text)
    return chunks

document_chunks = chunk_document_japanese(text)

def get_embeddings(texts):
    with torch.no_grad():
        inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
        outputs = embedder(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
    return embeddings

embeddings = get_embeddings(document_chunks)

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

system_message = (
    "あなたは「AIくん」というチャットボットアシスタントで、「神明工業"
    "という会社の従業員にサービスを提供しています"
    "あなたは以下の文書に基づいて正確な情報を提供します。"
    "従業員が文書に記載されていない情報を尋ねた場合は、"
    "正直に誠実に知らないと答えます。常に人々を尊重し、"
    "質問にはできるだけ詳しく答えてください。"
    "これがあなたのナレッジベースです。この「QUERY」に答えてください。"
)

def retrieve_and_generate(query, conversation_history=None, past_contexts=None, max_history_length=8):
    if past_contexts is None:
        past_contexts = set()
    if conversation_history is None:
        conversation_history = []

    query_embedding = get_embeddings([query])
    distances, indices = index.search(query_embedding, k=10)

    valid_results = [(idx, dist) for idx, dist in zip(indices[0], distances[0]) if idx != -1]
    ranked_chunks = sorted(valid_results, key=lambda x: x[1])
    
    retrieved_chunks = []
    total_tokens = 0
    max_retrieved_chunks = 10
    for idx, distance in ranked_chunks[:max_retrieved_chunks]:
        if idx not in past_contexts:
            chunk = document_chunks[idx]
            chunk_tokens = len(tokenizer(chunk)['input_ids'])
            retrieved_chunks.append((chunk, distance))
            total_tokens += chunk_tokens
            past_contexts.add(idx)
    
    context = system_message + '\n'.join(chunk for chunk, _ in retrieved_chunks)
    conversation_history.append(f"Query: {query}\n")
    context_with_history = context + '\n'.join(conversation_history)

    payload = {
        "model": "Qwen/Qwen2-72B-Instruct", 
        "messages": [
            {"role": "system", "content": system_message},
            *conversation_history,
            {"role": "user", "content": context_with_history}
        ],
        "temperature": 0,
        "frequency_penalty": 1,
        "presence_penalty": 1
    }

    response = requests.post(remote_model_url, json=payload)
    response.raise_for_status()
    
    answer = response.json()['choices'][0]['message']['content']

    conversation_history.append({"role": "user", "content": query})
    conversation_history.append({"role": "assistant", "content": answer})

    while len(tokenizer.encode(json.dumps(conversation_history))) > 10000:
        conversation_history.pop(0)

    torch.cuda.empty_cache()
    return context, query, answer, conversation_history

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(StarletteSessionMiddleware, secret_key="supersecretkey")

@app.post("/query")
async def query(request: Request):
    try:
        data = await request.json()
        query_text = data.get("query", "")
        session = request.session
        conversation_history = session.get("conversation_history", [])
        
        context, query, answer, conversation_history = retrieve_and_generate(query_text, conversation_history)
        session["conversation_history"] = conversation_history
        return {"context": context, "query": query, "answer": answer}
    except Exception as e:
        return {"error": str(e), "message": "Failed to process the query"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
