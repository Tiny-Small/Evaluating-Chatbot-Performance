# setup.py

from config import device, DB_FAISS_PATH, LLM_PATH, EMBEDDING_TYPE, EMBEDDING_FOLDER
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import LlamaCpp

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

def setup_embeddings():
    """Set up and return HuggingFace embeddings."""
    modelPath = EMBEDDING_FOLDER + EMBEDDING_TYPE

    embeddings = HuggingFaceEmbeddings(model_name=modelPath, model_kwargs={'device': device})
    return embeddings

def setup_database(embeddings):
    """Load and return the FAISS database using provided embeddings."""
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    return db

def setup_retriever(db):
    """Set up and return a retriever from the database."""
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    return retriever

def setup_llm():
    """Set up and return a configured LlamaCpp model."""
    callback_manager = CallbackManager([(StreamingStdOutCallbackHandler())])

    llm = LlamaCpp(
        model_path=LLM_PATH,
        n_gpu_layers=20,
        temperature=1,
        max_tokens=512,
        n_ctx=2048,
        n_batch=1024,
        repeat_penalty=1.2,
        callback_manager = callback_manager,
        verbose = True # Debugging
    )
    return llm

def setup_history_aware_retriever(llm, retriever):
    contextualize_q_system_prompt = "You are a summarizer. " \
                                    "Using available information, rephrase the human's follow-up question into a clear, standalone question. " \
                                    "This rephrased question should make sense on its own without any additional context from the chat history. " \
                                    "Your reply must contain only the standalone question. "
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Review the provided chat history:"),
            MessagesPlaceholder("chat_history"),
            ("system", "Review the human's follow-up question:"),
            ("human", "{input}"),
            ("system", contextualize_q_system_prompt)
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
    )

    return history_aware_retriever

def setup_question_answer_chain(llm):
    qa_system_prompt = "Your reply must contain only the answer based on the provided chat history. " \
                    "If the information is insufficient to formulate an answer, state that more information is needed. " \
                    "Do not attempt to impersonate a human or any specific system. Your reply must strictly be an automated response to the query based on the provided context."

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an automated chat assistant, not a human. Your task is to provide accurate and factual answers based solely on the chat history and context provided."),
            MessagesPlaceholder("chat_history"),
            ("system", "Review the context for the current conversation:"),
            ("system", "Context: {context}"),
            ("system", "Using only the information provided, generate a succinct and direct automated response to the following question from the human:"),
            ("human", "{input}"),
            ("system", qa_system_prompt)
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    return question_answer_chain

def setup_rag_chain(history_aware_retriever, question_answer_chain):
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    return rag_chain

def initialize_application():
    """Initialize all components and return them."""
    embeddings = setup_embeddings()
    db = setup_database(embeddings)
    retriever = setup_retriever(db)
    llm = setup_llm()
    history_aware_retriever = setup_history_aware_retriever(llm, retriever)
    question_answer_chain = setup_question_answer_chain(llm)
    rag_chain = setup_rag_chain(history_aware_retriever, question_answer_chain)
    return rag_chain

def setup_conversational_chain(rag_chain, session_history_getter):
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        session_history_getter,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
    return conversational_rag_chain


##############################
######### FORMATTING #########
##############################

def format_sources(sources):
    '''This is to format the source from the documents for easy viewing.'''
    formatted_sources = "Sources:\n"
    unique_sources = {source['page']: source['source'] for source in sources}.items()
    for page, source in unique_sources:
        formatted_sources += f"- {source}, Page {page}\n"
    return formatted_sources

def clean_response(response):
    # Identify the keywords to look for
    keywords = ["Assistant:", "AI:", "System:", "Automated Response:"]

    words = response.split()
    index = max((words.index(word) for word in words if word in keywords), default=None)

    # If a keyword is found, return the response from the word after the keyword onwards
    if index is not None:
        return ' '.join(words[index + 1:])
    else:
        # If no keyword is found, return the original response
        return response
