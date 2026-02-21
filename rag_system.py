import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma

load_dotenv()

class RAGSystem:
    def __init__(self, pdf_path="DH-Chapter2.pdf", persist_directory="./chroma_db"):
        self.pdf_path = pdf_path
        self.persist_directory = persist_directory
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        self.vector_store = None
        self.retriever = None
        self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

    def initialize(self):
        if os.path.exists(self.persist_directory):
            print(f"Loading existing vector store from {self.persist_directory}")
            self.vector_store = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
        else:
            print(f"Creating new vector store from {self.pdf_path}")
            loader = PyPDFLoader(self.pdf_path)
            documents = loader.load()
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100
            )
            chunks = text_splitter.split_documents(documents)
            
            self.vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=self.persist_directory
            )
        
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )

    def retrieve(self, query):
        if not self.retriever:
            self.initialize()
        return self.retriever.get_relevant_documents(query)

if __name__ == "__main__":
    rag = RAGSystem()
    rag.initialize()
    docs = rag.retrieve("What are the rules for passing a school bus?")
    for doc in docs:
        print(f"Source: {doc.metadata['page']}, Content: {doc.page_content[:100]}...")
