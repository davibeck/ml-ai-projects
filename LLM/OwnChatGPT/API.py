# save this as app.py
import os
import pinecone
from flask import Flask
from flask_cors import CORS, cross_origin
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
import base64

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

os.environ["OPENAI_API_KEY"] = "OpenAi API Key"

loader = DirectoryLoader('./Reports/')
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
docs = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()
query_result = embeddings.embed_query("Hello")

pinecone.init(
    api_key="Pinecone API Key",
    environment="us-west1-gcp-free"
)

index_name = "langchain-demo"

index = Pinecone.from_documents(docs, embeddings, index_name=index_name)
llm = OpenAI()
chain = load_qa_chain(llm, chain_type="stuff")

def get_similar_docs(query, k=2, score=False):
    if score:
        similar_docs = index.similarity_search_with_score(query, k=2)
    else:
        similar_docs = index.similarity_search(query, k=2)
    return similar_docs


def get_answer(query):
    similar_docs = get_similar_docs(query)
    answer = chain.run(input_documents=similar_docs, question = query)
    if (answer == "Não sei."):
        answer = "Para mais informações, acessar Site!"
    return answer

@app.route("/ask/<query>")
@cross_origin()
def ask(query):
    return get_answer(query)

if __name__ == '__main__':
    app.run()