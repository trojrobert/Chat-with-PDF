import os
import streamlit as st 
from authtoken import openai_api_key 
from pdf_processing import parse_pdf, clean_text, merge_hyphenated_words, fix_newlines, remove_multiple_newlines, create_chunks
from chat import create_chain

from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import HumanMessage, AIMessage

os.environ["OPENAI_API_KEY"] = openai_api_key

# App title 
st.title("Chat with your PDF 💬➜📄")
st.caption("Developed by Robert John")
# Create file uploader to read pdf file 
uploaded_file = st.file_uploader("Choose your .pdf file", type="pdf")

if uploaded_file is not None:

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
    vector_store = Chroma.from_documents(
        docment_chunks,
        embeddings,
        collection_name="pdf_collection",
        persist_directory="vector_store/chroma",
    )

    vector_store.persist()
    print(vector_store)

    chat_history = []
    chain = create_chain(collection_name=collection_name,
                    store_path=store_path)

    question = st.text_input("Ask your PDF a question")

    if question:
        response = chain({"question": question,
                        "chat_history": chat_history})
        answer = response['answer']
        source_documents = response['source_documents']
        st.write(f"Answer : {answer}")
        #st.write(f"Source : {source_documents}")

        chat_history.append(HumanMessage(content=question))
        chat_history.append(AIMessage(content=answer))