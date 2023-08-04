from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.embeddings import HuggingFaceEmbeddings
from langchain import HuggingFaceHub
from langchain.vectorstores import FAISS
from  langchain.schema import Document
from typing import Iterable
import os
import json


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


#Check for existing files and populate user_records dictionary
user_records = {}
for file_name in os.listdir():
    if file_name.endswith(".txt"):
        user_id = file_name.split("_", 1)[0]
        name = get_user_name(user_id)
        if name:
            user_records[user_id] = name

while True:
    user_id = input('Enter User ID: ')
    name = input('Enter User name: ')

    # Check if the user ID already exists
    if user_id in user_records:
        if user_records[user_id] == name:
            # User with the same ID and name exists, append chat to the existing file
            file_name = get_file_name(user_id, name)
            chat_history = []
            with open(file_name, "r") as file1:
                for line in file1:
                    query, answer = line.strip().split(' ', 1)
                    chat_history.append((query, answer))

            while True:
                query = input('Bot: ')
                if query.lower() in ["exit", "quit", "q"]:
                    print('Exiting')
                    break

                result = qa({'question': query, 'chat_history': chat_history})
                print('Answer: ' + result['answer'])
                chat_history.append((query, result['answer']))

            # Append the new chat history to the existing file
            with open(file_name, "a") as file1:
                for entry in chat_history[len(chat_history) - len(chat_history):]:
                    file1.write('question: "' + entry[0] + '" answer: "' + entry[1] + '"\n')
        else:
            print(f"User ID {user_id} is already used with a different name. Please enter a different ID or the same name.")

    else:
        chat_history = []
        while True:
            query = input('Bot: ')
            if query.lower() in ["exit", "quit", "q"]:
                print('Exiting')
                break

            result = qa({'question': query, 'chat_history': chat_history})
            print('Answer: ' + result['answer'])
            chat_history.append((query, result['answer']))

        # Append the chat history to the user-specific file or create a new file
        file_name = get_file_name(user_id, name)
        with open(file_name, "a") as file1:
            for entry in chat_history:
                file1.write('question: "' + entry[0] + '" answer: "' + entry[1] + '"\n')

        # Update the user_records dictionary with the new user entry
        user_records[user_id] = name

        print(f"Chat history for User ID {user_id} and name '{name}' has been saved in {file_name}.")

    # After each user session, ask if another user wants to continue or quit the chatbot
    user_choice = input("Do you want to chat with another user? (yes/no): ")
    if user_choice.lower() in ["no", "n", "quit", "q"]:
        print("Chatbot session ended. Goodbye!")
        break



# Additional code can be added here for further actions after all users finish using the chatbot.
# For example, closing any open connections or performing cleanup tasks.


print("memory:",memory)
print("chat_history",chat_history)
