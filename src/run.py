import os
import streamlit as st 
from authtoken import openai_api_key 
from langchain.document_loaders import PyPDFLoader
import datetime
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.chains import RetrievalQA


from pdf_processing import parse_pdf, clean_text, merge_hyphenated_words, fix_newlines, remove_multiple_newlines, create_chunks
from chat import create_chain
from langchain.chat_models import ChatOpenAI

from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
# from langchain.schema import HumanMessage, AIMessage


with st.sidebar:
    st.text("Enter your OpenAI API Key")
    st.session_state.OPENAI_API_KEY = st.text_input(label='*We do NOT store and cannot view your API key*',
                                                    placeholder='sk-p999HAfj6Cm1bO00SXgJc7kFxvFPtQ1KBBWrqSOU',
                                                    type="password",
                                                    help='You can find your Secret API key at \
                                                            https://platform.openai.com/account/api-keys')
    
    




# App title 
st.title("Chat with your PDF 💬➜📄")
st.caption("Developed by Robert John")



# Create file uploader to read pdf file 
uploaded_file = st.file_uploader("Choose your .pdf file", type="pdf")

if uploaded_file is not None:

    if not st.session_state.OPENAI_API_KEY:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()

    os.environ["OPENAI_API_KEY"] = st.session_state.OPENAI_API_KEY
    #openai.api_key = openai_api_key

    current_date = datetime.datetime.now().date()
    if current_date < datetime.date(2023, 9, 2):
        llm_name = "gpt-3.5-turbo-0301"
    else:
        llm_name = "gpt-3.5-turbo"
    print(llm_name)
    llm = ChatOpenAI(model_name=llm_name, temperature=0)
    


    # Read the pdf file and parse it
    metadata, pages = parse_pdf(uploaded_file)


    # Create a list of functions that will be used for cleaning the text 
    cleaning_functions =  [merge_hyphenated_words,
                            fix_newlines,
                            remove_multiple_newlines]
    # clean text in pages 
    cleaned_pages = clean_text(pages, cleaning_functions)
    
    # Divide the clean PDF file into chunks 
    docment_chunks = create_chunks(cleaned_pages, metadata)

    # Convert the chunks to embeddings
    embeddings = OpenAIEmbeddings()

    collection_name = "pdf_collection"
    store_path = "vector_store/chroma"
    vectordb = Chroma.from_documents(
        documents=docment_chunks,
        persist_directory=store_path, 
        embedding=embeddings)
    vectordb.persist()


    # vector_store = Chroma.from_documents(
    #     docment_chunks,
    #     embeddings,
    #     collection_name="pdf_collection",
    #     persist_directory="vector_store/chroma",
    # )

    # vector_store.persist()

    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vectordb.as_retriever()
    )

    question = st.text_input("Ask your PDF a question")

    if question:
        #docs = vectordb.similarity_search(question, k=3)
        result = qa_chain({"query": question})
        print(f"######## {result['result']}")
        st.write(result['result'])

    # print(vector_store)

    # chat_history = []
    # chain = create_chain(collection_name=collection_name,
    #                 store_path=store_path)

    # question = st.text_input("Ask your PDF a question")

    # if question:
    #     response = chain({"question": question,
    #                     "chat_history": chat_history})
    #     answer = response['answer']
    #     source_documents = response['source_documents']
    #     st.write(f"Answer : {answer}")
    #     #st.write(f"Source : {source_documents}")

    #     chat_history.append(HumanMessage(content=question))
    #     chat_history.append(AIMessage(content=answer))