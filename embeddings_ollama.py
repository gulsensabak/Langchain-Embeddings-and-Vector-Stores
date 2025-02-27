from langchain_ollama import OllamaEmbeddings

# create the embedding model
# we need to pass the model name. Llama model can be act as an embedding model as well. Llama is a multimodal model.
llm = OllamaEmbeddings(model = "llama3.2:latest")

# get the input from user
text = input("Enter the text: ")

# create embedding vector
response = llm.embed_query(text)

print(response)