# from langchain.text_splitter import CharacterTextSplitter
# from langchain_community.embeddings.spacy_embeddings import SpacyEmbeddings
# from langchain_community.vectorstores import FAISS
# from PyPDF2 import PdfReader
# from langchain_groq import ChatGroq
# from langchain_core.prompts import ChatPromptTemplate
# from langchain.agents import AgentExecutor, create_tool_calling_agent
# from langchain.tools.retriever import create_retriever_tool
# import os
# import groq
# import streamlit as st

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Reading the file
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file is not None:
    pdf_reader = PdfReader(uploaded_file)
    raw_text = ""
    for page in pdf_reader.pages:
        raw_text += page.extract_text()

    # Create Chunks of Data
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )

    text_chunks = text_splitter.split_text(raw_text)

    # Vector Embedding of Chunks
    embeddings = SpacyEmbeddings(model_name="en_core_web_sm")
    vector_storage = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    vector_storage.save_local("faiss_db")

    db = FAISS.load_local("faiss_db", embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever()
    retrieval_chain = create_retriever_tool(retriever, "pdf_reader", "It is a tool to read data from pdfs")

    # LLM
    llm = ChatGroq(temperature=0, model="llama3-70b-8192", api_key="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")

    # Prompt Template
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a helpful assistant. Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
        provided context just say, "answer is not available in the context", don't provide the wrong answer""",
            ),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )

    # Agent
    tool = [retrieval_chain]
    agent = create_tool_calling_agent(llm, tool, prompt)
    agent_executer = AgentExecutor(agent=agent, tools=tool, verbose=True)

    st.title("PDF Query Assistant")
    user_input = st.text_input("Enter your question:")

    if st.button("Find Answer"):
        response = agent_executer.invoke({"input": user_input})
        st.write(f"Input: {response['input']}")
        st.write(f"Output: {response['output']}")
