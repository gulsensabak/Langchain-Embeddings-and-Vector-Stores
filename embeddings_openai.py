import os
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

# get the api key from .env
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# create embeding model
# since OpenAIEmbeddings brings the specific embedding model in it, we dont need to specify the model
llm = OpenAIEmbeddings(api_key = OPENAI_API_KEY)

# you can calculate the embeddings of this text
text = input("Enter the text")

# embed_query function will calculate the embeddings and it will get a response back
response = llm.embed_query(text)

# response itself is a list of vectors which represent the embeddings for the given text
print(response)