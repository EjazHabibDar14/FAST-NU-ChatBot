import json
import os
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import Replicate
from langchain.vectorstores import FAISS
from  langchain.schema import Document
from typing import Iterable
from fastapi import FastAPI, HTTPException 
from pydantic import BaseModel
from typing import List
import re
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi import Response
from langchain.llms import Replicate

def load_docs_from_jsonl(file_path)->Iterable[Document]:
    array = []
    with open(file_path, 'r') as jsonl_file:
        for line in jsonl_file:
            data = json.loads(line)
            obj = Document(**data)
            array.append(obj)
    return array

chunked_docs=load_docs_from_jsonl('data.jsonl')
#print(chunked_docs)

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

# Create embeddings and store them in a FAISS vector store
embedder = HuggingFaceEmbeddings()
vector_store = FAISS.from_documents(chunked_docs, embedder)
print(vector_store)

from getpass import getpass

REPLICATE_API_TOKEN = getpass()

#os.environ["REPLICATE_API_TOKEN"] = "Your Replicate Token goes here"

llm = Replicate(
     model="lucataco/llama-2-7b-chat:6ab580ab4eef2c2b440f2441ec0fc0ace5470edaf2cbea50b8550aec0b3fbd38",
     input={"temperature": 0.75, "max_length": 500, "top_p": 1},
 )

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

qa = ConversationalRetrievalChain.from_llm(
    llm=llm, retriever=vector_store.as_retriever(), memory=memory)

def get_file_name(user_id, name):
    return f"{user_id}_{name}.txt"

def user_exists(user_id, name):
    file_name = get_file_name(user_id, name)
    return os.path.exists(file_name)

def get_user_name(user_id):
    for file_name in os.listdir():
        if file_name.startswith(f"{user_id}_"):
            return file_name.split("_", 1)[1].split(".")[0]
    return None

app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Adjust this to your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatHistoryItem(BaseModel):
    user_id: str
    user_name: str
    question: str
    answer: str

def load_chat_history(file_path: str) -> List[ChatHistoryItem]:
    chat_history = []
    user_id = file_path.split("_", 1)[0]
    name = get_user_name(user_id)
    with open(file_path, "r") as file1:
        for line in file1:
            line = line.strip()
            if len(line) > 0:
                # Use regular expressions to extract question and answer between double quotes
                match = re.match(r'question: "(.*?)" answer: "(.*?)"', line)
                if match:
                    question = match.group(1)
                    answer = match.group(2)
                    chat_history.append(ChatHistoryItem(user_id=user_id, user_name=name, question=question, answer=answer))
                else:
                    # Handle the case where the line does not match the expected format
                    print(f"Invalid line in file '{file_path}': {line}")
    return chat_history

# from fastapi.templating import Jinja2Templates
# templates = Jinja2Templates(directory="templates")


# @app.get("/")
# async def index(request: Request):
#     return templates.TemplateResponse("index.html", {"request": request})

@app.get("/chat_me/", response_model=List[ChatHistoryItem])
async def get_all_chat_history():
    all_chat_history = []
    for file_name in os.listdir():
        if file_name.endswith(".txt"):
            all_chat_history.extend(load_chat_history(file_name))
    return all_chat_history

@app.get("/chat_me/{user_id}", response_model=List[ChatHistoryItem])
async def get_user_chat_history(user_id: str):
    user_chat_history = []
    for file_name in os.listdir():
        if file_name.endswith(".txt") and file_name.startswith(f"{user_id}_"):
            user_chat_history.extend(load_chat_history(file_name))
    return user_chat_history

@app.get("/check_user/{user_id}/{name}", response_model=dict)
def check_user(user_id: str, name: str):
    actual_id = get_user_name(user_id)

    if actual_id is None:
        # New id, user ID doesn't exist
        return {"exists": False, "idMatches": False, "canChat": True}
    
    if actual_id == name:
        # id exists and name matches
        return {"exists": True, "idMatches": True, "canChat": True}
    else:
        # id exists but name doesn't match
        return {"exists": True, "idMatches": False, "canChat": False}




class ChatRequest(BaseModel):
    user_id: str
    name: str
    question: str

class ChatResponse(BaseModel):
    answer: str

def load_user_records():
    user_records = {}
    for file_name in os.listdir():
        if file_name.endswith(".txt"):
            user_id, name = file_name.split("_", 1)
            name = name.split(".")[0]
            user_records[user_id] = name
    return user_records

# Check for existing files and populate user_records dictionary
user_records = load_user_records()

@app.post("/Chat_me", response_model=ChatResponse)
def chat_me(request: ChatRequest):
    user_id = request.user_id
    name = request.name
    question = request.question

    if question.lower() == "quit":
        return Response(content="You have left the chat. Type 'Start Chat' again to start a new chat session.", media_type="text/plain")
    # Check if the user ID already exists
    if user_id in user_records:
        if user_records[user_id] == name:
            # User with the same ID and name exists, append chat to the existing file
            file_name = get_file_name(user_id, name)
            chat_history = load_chat_history(file_name)
        else:
            # User ID exists but with a different name, return an error response
            raise HTTPException(status_code=400, detail="User ID already belongs to another user.")
    else:
        # New user, create a new file and add to user_records dictionary
        chat_history = []
        user_records[user_id] = name

    result = qa({'question': question, 'chat_history': chat_history})
    answer = result['answer']
    chat_history.append((question, answer))

    # Append the chat history to the user-specific file or create a new file
    file_name = get_file_name(user_id, name)
    with open(file_name, "a") as file1:
        file1.write('question: "' + question + '" answer: "' + answer + '"\n')

    return JSONResponse(content={"response": answer})
