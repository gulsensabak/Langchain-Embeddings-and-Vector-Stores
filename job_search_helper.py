# vector store allows us to store and retrieve data using high dimensional vectors
import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from dotenv import load_dotenv

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
db = Chroma.from_documents(chunks, llm)

# get the input from user
text = input("Enter the query: ")

# embed the input taken from user
embedding_vector = llm.embed_query(text)
 
# search based on embeddings similarity not keyword similarity
docs = db.similarity_search_by_vector(embedding_vector)

# print all the content of related embeddings
for doc in docs:
    print(doc.page_content)