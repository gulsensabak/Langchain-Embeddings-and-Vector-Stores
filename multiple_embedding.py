from langchain_openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv

# get the api key
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# create embedding model
embeddings = OpenAIEmbeddings(api_key= OPENAI_API_KEY)

# create embedding
# embed_documents function calculate the embeddings for multiple text values at once
response = embeddings.embed_documents(
    [
        "I love playing video games",
        "I am going to the movie",
        "I love coding",
        "Hello World!"
    ]
)

# to check whether it creates the embeddings correctly or not
print(len(response))

# to reach the embedding vector of the first sentence
print(response[0])