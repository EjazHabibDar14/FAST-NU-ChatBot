# import requests
# import sys
import json
import os
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain import HuggingFaceHub
from  langchain.schema import Document
from typing import Iterable
from fastapi import FastAPI, HTTPException 
from pydantic import BaseModel
from typing import List, Tuple
from fastapi.staticfiles import StaticFiles
import re
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


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
#print(vector_store)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

repo_id="google/flan-t5-small"
qa = ConversationalRetrievalChain.from_llm(
    HuggingFaceHub(
    repo_id=repo_id, model_kwargs={"temperature": 0.5, "max_length":512, "min_length": 8}
    ), vector_store.as_retriever(), memory=memory)

def get_file_name(user_id, name):
    return f"{user_id}_{name}.txt"

# Function to check if a user with the given ID and name already exists
def user_exists(user_id, name):
    file_name = get_file_name(user_id, name)
    return os.path.exists(file_name)

# Function to get the user's name based on the user ID
def get_user_name(user_id):
    for file_name in os.listdir():
        if file_name.startswith(f"{user_id}_"):
            return file_name.split("_", 1)[1].split(".")[0]
    return None

# Check for existing files and populate user_records dictionary
# user_records = {}
# for file_name in os.listdir():
#     if file_name.endswith(".txt"):
#         user_id = file_name.split("_", 1)[0]
#         name = get_user_name(user_id)
#         if name:
#             user_records[user_id] = name

# while True:
#     user_id = input('Enter User ID: ')
#     name = input('Enter User name: ')

#     # Check if the user ID already exists
#     if user_id in user_records:
#         if user_records[user_id] == name:
#             # User with the same ID and name exists, append chat to the existing file
#             file_name = get_file_name(user_id, name)
#             chat_history = []
#             with open(file_name, "r") as file1:
#                 for line in file1:
#                     query, answer = line.strip().split(' ', 1)
#                     chat_history.append((query, answer))

#             while True:
#                 query = input('Bot: ')
#                 if query.lower() in ["exit", "quit", "q"]:
#                     print('Exiting')
#                     break

#                 result = qa({'question': query, 'chat_history': chat_history})
#                 print('Answer: ' + result['answer'])
#                 chat_history.append((query, result['answer']))

#             # Append the new chat history to the existing file
#             with open(file_name, "a") as file1:
#                 for entry in chat_history[len(chat_history) - len(chat_history):]:
#                     file1.write('question: "' + entry[0] + '" answer: "' + entry[1] + '"\n')
#         else:
#             print(f"User ID {user_id} is already used with a different name. Please enter a different ID or the same name.")

#     else:
#         chat_history = []
#         while True:
#             query = input('Bot: ')
#             if query.lower() in ["exit", "quit", "q"]:
#                 print('Exiting')
#                 break

#             result = qa({'question': query, 'chat_history': chat_history})
#             print('Answer: ' + result['answer'])
#             chat_history.append((query, result['answer']))

#         # Append the chat history to the user-specific file or create a new file
#         file_name = get_file_name(user_id, name)
#         with open(file_name, "a") as file1:
#             for entry in chat_history:
#                 file1.write('question: "' + entry[0] + '" answer: "' + entry[1] + '"\n')

#         # Update the user_records dictionary with the new user entry
#         user_records[user_id] = name

#         print(f"Chat history for User ID {user_id} and name '{name}' has been saved in {file_name}.")

#     # After each user session, ask if another user wants to continue or quit the chatbot
#     user_choice = input("Do you want to chat with another user? (yes/no): ")
#     if user_choice.lower() in ["no", "n", "quit", "q"]:
#         print("Chatbot session ended. Goodbye!")
#         break



# # Additional code can be added here for further actions after all users finish using the chatbot.
# # For example, closing any open connections or performing cleanup tasks.


# print("memory:",memory)
# print("chat_history",chat_history)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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

@app.post("/chat_me/", response_model=ChatResponse)
async def chat_with_user(request: ChatRequest):
    user_id = request.user_id
    name = request.name
    question = request.question

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

    result = qa({'question': question, 'chat_history': chat_history})
    answer = result['answer']
    chat_history.append((question, answer))

    # Append the chat history to the user-specific file or create a new file
    file_name = get_file_name(user_id, name)
    with open(file_name, "a") as file1:
        file1.write('question: "' + question + '" answer: "' + answer + '"\n')

    # Update the user_records dictionary with the new user entry
    user_records[user_id] = name

    return ChatResponse(answer=answer)