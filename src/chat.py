from langchain.vectorstores.chroma import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain



def create_chain(collection_name, store_path):

    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.0,
    )

    embedding = OpenAIEmbeddings()

    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embedding,
        persist_directory=store_path, 
    )

    chain =  ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        return_source_documents=True,
    )

    return chain 
