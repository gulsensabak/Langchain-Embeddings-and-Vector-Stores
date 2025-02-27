import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# get the api key
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# create embedding model
llm = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

# get the document
document = TextLoader("job_listings.txt").load()

# create spliter to the text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 200, chunk_overlap = 10)

# split the text into chunks
chunks = text_splitter.split_documents(document)

# Chroma is the vector store 
# pass the chunks to the embedding model, after creating embeddings store them in a vector store (called db)
db = FAISS.from_documents(chunks, llm)

# taking plain text, calculate the embedding for it, then query the vector database with that embedded text and retrieve the data for us 
retriever = db.as_retriever()

# get the input from user
text = input("Enter the query: ")
 
# pass plain text to the invoke method as a parameter so that retriever could use it
docs = retriever.invoke(text)

# print all the content of related embeddings
for doc in docs:
    print(doc.page_content)