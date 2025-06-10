import streamlit as st
import time
import query_engine 

st.set_page_config(page_title="Chat UI Test", layout="wide")
st.title("Chat Assistant UI")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for role, msg in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(msg)

user_chat, context_window = st.columns([1.5, 1.5])

global context
load_context = False

with user_chat:
    user_input = st.chat_input("Ask your question:")

    if user_input:
        st.session_state.chat_history.append(("user", user_input))
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            with st.spinner("Thinking...", show_time=True):
                full_response, context = query_engine.query_rag(user_input)
            load_context = True

            displayed_text = ""
            for char in full_response:
                displayed_text += char
                response_placeholder.markdown(displayed_text + "▌")  
                time.sleep(0.001) 

            response_placeholder.markdown(displayed_text)
            st.session_state.chat_history.append(("assistant", full_response))

with context_window:
    if load_context:
        st.markdown("## [ Context Window ]")
        st.markdown(context)
