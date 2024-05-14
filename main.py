# main.py

from setup import setup_conversational_chain, initialize_application, format_sources, clean_response
from session_management import get_session_history
import streamlit as st
from datetime import datetime

#########################
####### Llama-Cpp #######
#########################

rag_chain = initialize_application()
session_history_getter = lambda session_id: get_session_history(session_id)

conversational_rag_chain = setup_conversational_chain(rag_chain, session_history_getter)

#########################
####### Streamlit #######
#########################

st.title("AI Q&A Bot")

# Ensure messages in the session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Ensure chat history in the session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        st.caption(message["timestamp"])

# React to user input using the conversational AI chain
user_input = st.chat_input("Type your question...")

if user_input:
    # Display user message
    st.chat_message("user").markdown(user_input)
    send_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.caption(send_time)

    # Add user message to chat history with timestamp
    st.session_state.messages.append({"role": "user", "content": user_input, "timestamp": send_time})

    # Invoke the conversational chain to get a response
    response = conversational_rag_chain.invoke(
        {"input": user_input, "chat_history": st.session_state.chat_history},
        config={"configurable": {"session_id": "-"}}  # session ID example
    )

    answer = response.get("answer", "No answer generated.")

    if answer !=  "No answer generated.":
        answer = clean_response(answer)

    context = response.get("context", [])
    if context:
        sources = [c.metadata for c in context]
        sources = format_sources(sources)
        answer += " " + sources

    # Display assistant response
    with st.chat_message("assistant"):
        st.markdown(answer)
        receive_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.caption(receive_time)

    # Add assistant response to chat history with timestamp
    st.session_state.messages.append({"role": "assistant", "content": answer, "timestamp": receive_time})

    # Add user message to chat history
    st.session_state.chat_history.extend([("human", user_input), ("ai", answer)])
