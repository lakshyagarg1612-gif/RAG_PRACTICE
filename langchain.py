import os 
from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader,PyPDFLoader,TextLoader,Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from google import genai

DATA_DIR="data"
documents=[]
documents+=DirectoryLoader(DATA_DIR,glob="**/*.pdf",loader_cls=PyPDFLoader).load()
documents+=DirectoryLoader(DATA_DIR,glob="**/*.txt",loader_cls=TextLoader,loader_kwargs={"encoding":"utf-8"}).load()
documents+=DirectoryLoader(DATA_DIR,glob="**/*.docx",loader_cls=Docx2txtLoader).load()
print(f"No. of files: {len(documents)}")

text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=100)
chunks=text_splitter.split_documents(documents)
print("Created Chunks")

CHORMA_DB="chroma_db"
embeddingmodel="sentence-transformers/all-MiniLM-L6-v2"
embedding_model=HuggingFaceEmbeddings(model_name=embeddingmodel)
if Path(CHORMA_DB).exists():
    print(f"Chroma Directory already exist")
    vector_db=Chroma(persist_directory=CHORMA_DB,embedding_function=embedding_model,)
else:
    vector_db=Chroma.from_documents(documents=chunks,embedding=embedding_model,persist_directory=CHORMA_DB,)
    print("Vector Created..")




TOP_K=3
load_dotenv()
api_key=os.getenv("GOOGLE_API")
if not api_key:
    raise ValueError("ERROR")
client=genai.Client(api_key=api_key)
while True:
    user_question=input("Ask questionL:").strip()
    if user_question.lower() in["exist","quit"]:
        print("Stopped")
        break
    
    retriver=vector_db.as_retriever(search_kwargs={"k":TOP_K})
    retrivered_docs=retriver.invoke(user_question)

    context="\n\n".join(doc.page_content for doc  in retrivered_docs)

    sys_prompt=f'''You are a Senior Advisor you have to give the response in more technical response by using the context efficiently
        this is user query {user_question} and this is context {context} you have to give response in small general explanation
        with some key important details'''
    reponse=client.models.generate_content(model="gemini-3.1-flash-lite",contents=sys_prompt,)
    print("Answer: ")
    print(reponse.text)
        