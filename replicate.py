import json
import os
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import Replicate
from langchain.vectorstores import FAISS
from langchain import HuggingFaceHub
from  langchain.schema import Document
from typing import Iterable
from fastapi import FastAPI, HTTPException 
from pydantic import BaseModel
from typing import List, Tuple
from fastapi.staticfiles import StaticFiles
import re
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi import Response


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

from getpass import getpass

REPLICATE_API_TOKEN = getpass()

os.environ["REPLICATE_API_TOKEN"] = "r8_1hYtsePHIjbil1g4h4Mmam70XutDPEx0LzhmU"

llm = Replicate(
    model="lucataco/llama-2-7b-chat:6ab580ab4eef2c2b440f2441ec0fc0ace5470edaf2cbea50b8550aec0b3fbd38",
    input={"temperature": 0.75, "max_length": 500, "top_p": 1},
)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)


qa = RetrievalQA.from_chain_type(llm=llm, 
                                 chain_type="stuff", 
                                 retriever=vector_store.as_retriever())
query = "Provide me with the details of Admission process at fast"
qa.run(query)