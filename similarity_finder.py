# in this file we will try to figure out how similar given two words or sentences are using their embedding values

from langchain_openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv
import numpy as np

# get the api key
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# create the embedding model
llm = OpenAIEmbeddings(api_key = OPENAI_API_KEY)

# get the texts from user
text1 = input("Enter the text1: ")
text2 = input("Enter the text2: ")

# get embedding vectors and put it to the response variables
response1 = llm.embed_query(text1)
response2 = llm.embed_query(text2)

# dot: take multiple embeddings and calculate the vector similarity (dot method uses mathematical cosine function [cosine function can calculate the similarity between vectors])
# it will give similarity score between 0 and 1. The higher the score the closer the texts are
similarity_score = np.dot(response1, response2)

# in tat way we can see how semantically similar the text1 and text2
print(similarity_score)