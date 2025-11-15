import os
import requests
import streamlit as st

CHAT_SERVER_URL = os.getenv("CHAT_SERVER_URL", "http://localhost:8000")

st.set_page_config(page_title="AI Agent Playground", page_icon="🤖")
st.title("AI Agent Playground")

if "session_id" not in st.session_state:
    import uuid
    st.session_state["session_id"] = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state["messages"] = []

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("メッセージを入力してください"):
    # ユーザーメッセージをローカルにも保存
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # chat_server に投げるペイロード
    payload = {
        "session_id": st.session_state["session_id"],
        "messages": st.session_state["messages"],
    }

    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("Thinking...")

        try:
            res = requests.post(f"{CHAT_SERVER_URL}/chat", json=payload, timeout=60)
            res.raise_for_status()
            data = res.json()
            reply = data.get("reply", "(no reply)")
        except Exception as e:
            reply = f"Error: {e}"

        placeholder.markdown(reply)
        st.session_state["messages"].append({"role": "assistant", "content": reply})