import streamlit as st

from langchain_community.document_loaders import PDFPlumberLoader
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import OllamaLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate


prompt_template = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Context: {context}
Question: {question}
Answer: 
"""

pdf_path = "Ollama-SLM/files/"
# initializes the component responsible for converting text into numerical vectors
embeddings = OllamaEmbeddings(model="llama3.2")
# initializes a vector database, which is used to store and search your text data based on its meaning
vectorstore = InMemoryVectorStore(embeddings=embeddings)
# initializes the LLM component
model = OllamaLLM(model="llama3.2")

# this is designed to handle a file object, that is uploaded by a user through web application (streamlit)
def upload_pdf(file):
    with open(pdf_path + file.name, "wb") as f:
        f.write(file.getbuffer()) # read the entire file object into a block of memory 

# this read the content of the uploaded pdf file, and structure it into a format usable by LLM pipeline
def load_pdf(file_path):
    loader = PDFPlumberLoader(file_path)
    documents = loader.load()
    return documents

def split_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000, # max size of characters
        chunk_overlap = 200, # size of overlap between chunks
        length_function = len, # function to calculate length of the chunk
        add_start_index = True) # include the start index of the chunk
    return text_splitter.split_documents(documents)

'''
the parameters within the split_text function, are for RAG (Retrieval Augmented Generation), which 
the chunk size is chosen to be small enough to fit within the LLM's context window but large enough to contain sufficient meaning; 
the overlap ensures that semantically relevant information at the beginning or end of one chunk is also present in the adjacent chunk, preventing loss of context;
and for tracing the chunk back to its source location in the original document, which can be useful when citing sources in the final answer.
'''

def index_docs(documents):
    vector_store.add_documents(documents) 
'''
here the documents are already split into chunks, and vector_store converts the text content of each chunk into a high-dimensional vector,
then store the vector and rhe original text chunk in the vector store.
'''

def retrieve_docs(query):
    return vector_store.similarity_search(query)
'''
this function takes a query as input and returns the most similar documents from the vector store based on semantic similarity.
'''
# generate answer
def answer_question(question, documents):
    context = "\n\n".join([doc.page_content for doc in documents])
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model
    return chain.invoke({"question": question, "context": context})

uploaded_file = st.file_uploader(
    "Upload PDF",
    type="pdf",
    accept_multiple_files=False
)

if uploaded_file:
    upload_pdf(uploaded_file)
    documents = load_pdf(pdfs_directory + uploaded_file.name)
    chunked_documents = split_text(documents)
    index_docs(chunked_documents)

    question = st.chat_input()

    if question:
        st.chat_message("user").write(question)
        related_documents = retrieve_docs(question)
        answer = answer_question(question, related_documents)
        st.chat_message("assistant").write(answer)