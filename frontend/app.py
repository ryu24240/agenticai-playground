import os, uuid
import requests
import streamlit as st

CHAT_SERVER_URL = os.getenv("CHAT_SERVER_URL", "http://localhost:8000")

st.set_page_config(page_title="AI Agent Playground", page_icon="🤖")
st.title("AI Agent Playground")

def render_selector():
    st.title("AI Agent Playground - Setup")

    orchestrator = st.selectbox(
        "Select Orchestrator",
        ["Semantic Kernel", "LangGraph"],
        index=0,
    )

    model = st.selectbox(
        "Select Model",
        ["llama", "qwen"],
        index=0,
    )

    if st.button("Start Playground"):
        st.session_state["orchestrator"] = orchestrator
        st.session_state["model"] = model
        # 次の画面を表示するフラグ
        st.session_state["setup_done"] = True
        st.rerun()

def render_playground():
    orchestrator = st.session_state.get("orchestrator", "Semantic Kernel")
    model = st.session_state.get("model", "llama")

    st.caption(f"Orchestrator: {orchestrator} / Model: {model}")

    # セッションID
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = str(uuid.uuid4())

    # メッセージ履歴
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # これまでの履歴を表示
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # ユーザー入力
    if prompt := st.chat_input("メッセージを入力してください"):
        # 1. ユーザーメッセージを履歴に追加・表示
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 2. chat_server に投げるペイロード
        payload = {
            "session_id": st.session_state["session_id"],
            "messages": st.session_state["messages"],
            "orchestrator": st.session_state.get("orchestrator"),
            "model": st.session_state.get("model"),
        }

        # 3. アシスタント側のプレースホルダ
        with st.chat_message("assistant"):
            placeholder = st.empty()
            placeholder.markdown("Thinking...")

            try:
                res = requests.post(f"{CHAT_SERVER_URL}/chat", json=payload, timeout=180)
                res.raise_for_status()
                data = res.json()
                reply = data.get("reply", "(no reply)")
            except Exception as e:
                reply = f"Error: {e}"

            # 4. 返信の表示 & 履歴追加
            placeholder.markdown(reply)
            st.session_state["messages"].append({"role": "assistant", "content": reply})

def main():
    if "setup_done" not in st.session_state:
        st.session_state["setup_done"] = False

    if not st.session_state["setup_done"]:
        render_selector()
    else:
        render_playground()

if __name__ == "__main__":
    main()

