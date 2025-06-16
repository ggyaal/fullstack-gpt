import streamlit as st
from openai import OpenAI
from langchain.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper
from langchain.utilities.wikipedia import WikipediaAPIWrapper
import requests
from bs4 import BeautifulSoup
import json
import os
import time
import datetime

EXTRACT_FILE_DIR = "./files"

def search_duckduckgo(inputs):
    query = inputs.get("query")
    ddg = DuckDuckGoSearchAPIWrapper()
    return ddg.run(query)

def search_wikipedia(inputs):
    query = inputs.get("query")
    wiki = WikipediaAPIWrapper()
    return wiki.run(query)

def website_scraping(inputs):
    url = inputs.get("url")
    try:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            return soup.get_text()
        return "No content found."
    except Exception as e:
        return f"Error extracting content: {str(e)}"

def save_or_append_to_text(inputs):
    file_name = inputs.get("file_name")
    text = inputs.get("text")
    file_path = f"{EXTRACT_FILE_DIR}/{file_name}"
    timestamp = datetime.datetime.now().strftime("[ %Y-%m-%d %H:%M:%S ]")
    try:
        with open(file_path, "a", encoding="utf-8") as file:
            file.write(f"----\n{timestamp}\n\n{text.strip()}\n\n")

        return f"""
You can view the file by selecting {file_name} in the Extracted Files section of the sidebar.

Extracted Text: {text.strip()}
(Full details available in the sidebar.)
        """
    except Exception as e:
        return f"Error saving content: {str(e)}"

functions_map = {
    "search_duckduckgo": {"func": search_duckduckgo, "status": "🦆 DuckDuckGo 검색 중 ..."},
    "search_wikipedia": {"func": search_wikipedia, "status": "📜 Wikipedia 검색 중 ..."},
    "website_scraping": {"func": website_scraping, "status": "🌐 웹사이트 스크래핑 중 ..."},
    "save_or_append_to_text": {"func": save_or_append_to_text, "status": "💾 텍스트 저장 중 ..."},
}

client = None

def send_message(thread_id, query):
    return client.beta.threads.messages.create(
        thread_id=thread_id, role="user", content=query
    )

def handle_tool_calls(tool_calls):
    outputs = []

    for tool_call in tool_calls:
        tool_id = tool_call.id
        function = tool_call.function
        outputs.append(
            {
                "output": functions_map[function.name](json.loads(function.arguments)),
                "tool_call_id": tool_id,
            }
        )
    return outputs

def create_thread(query):
    thread = client.beta.threads.create(
        messages=[
            {
                "role": "user",
                "content": query,
            }
        ]
    )
    st.session_state["thread_id"] = thread.id
    return thread.id

def get_thread_id():
    if "thread_id" not in st.session_state:
        st.session_state["thread_id"] = None
    return st.session_state["thread_id"]

def save_message(role, message):
    thread_id = get_thread_id()
    if "messages" not in st.session_state:
        st.session_state["messages"] = {}
    if thread_id is not None:
        if thread_id not in st.session_state["messages"]:
            st.session_state["messages"][thread_id] = []
        st.session_state["messages"][thread_id].append({
            "role": role,
            "content": message
        })

def send_message(role, message, is_save=True):
    with st.chat_message(role):
        st.markdown(message)
    if is_save:
        save_message(role, message)

def paint_history():
    thread_id = get_thread_id()
    if "messages" not in st.session_state:
        st.session_state["messages"] = {}
    for msg in st.session_state["messages"].get(thread_id, []):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

def get_selected_file():
    if "selected_file" not in st.session_state:
        st.session_state["selected_file"] = None
    return st.session_state["selected_file"]

def get_page_number():
    if "file_page" not in st.session_state:
        st.session_state["file_page"] = 0
    return st.session_state["file_page"]

def get_extract_files(page=0):
    extract_dir = EXTRACT_FILE_DIR
    if not os.path.exists(extract_dir):
        os.makedirs(extract_dir)

    size = 5
    files = [ f for f in os.listdir(extract_dir) if os.path.isfile(os.path.join(extract_dir, f)) ]
    limit = page * size

    return { "files": files[limit:limit+size], "is_last": limit + size >= len(files) }

def get_selected_file_content(file_name):
    file_path = f"{EXTRACT_FILE_DIR}/{file_name}"
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()
    return None

st.set_page_config(
    page_title="Assistants",
    page_icon=":mag:",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.title("Assistants")

with st.sidebar:
    assistant_id = "asst_YENZp7WmMQlv6MnX2XcMhx0e"
    openai_api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        placeholder="Enter your OpenAI API key",
        key="openai_api_key",
    )

extract_files = get_extract_files(get_page_number())

with st.sidebar.expander("📂 Extracted Files"):
    for file in extract_files["files"]:
        if st.button(label=file.replace(".txt", ""), key=file, use_container_width=True):
            st.session_state["selected_file"] = file
    prev, page_num, next = st.columns([2, 1, 2])
    if prev.button("이전", key="prev_file", use_container_width=True, disabled=get_page_number() == 0):
        st.session_state["file_page"] = max(0, get_page_number() - 1)
        st.rerun()
    page_num.button(str(get_page_number() + 1), key="file_page_num", use_container_width=True, disabled=True)
    if next.button("다음", key="next_file", use_container_width=True, disabled=extract_files["is_last"]):
        st.session_state["file_page"] = st.session_state.get("file_page", 0) + 1
        st.rerun()
    if get_selected_file():
        if st.button("돌아가기", key="reset_file_page", use_container_width=True, type="primary"):
            st.session_state["selected_file"] = None
            st.rerun()

if get_selected_file():
    selected_file = get_selected_file()
    st.markdown(f"""
## {selected_file.replace(".txt", "")}
{get_selected_file_content(selected_file) or "No content available."}
    """)
else:
    if not openai_api_key:
        st.markdown(
            """
## Assistants

궁금한 걸 물어보세요.
AI가 `DuckDuckGo`와 `Wikipedia`를 동시에 검색해서
핵심 정보를 추출하고 필요하다면 추출해서 저장합니다!

모은 정보는 왼쪽 사이드바의 📂 Extracted Files에서 클릭 한 번으로 열 수 있어요.
            """
        )
        selected_file = None
    else:
        client = OpenAI(api_key=openai_api_key)
        send_message("assistant", "안녕하세요! 무엇을 검색할까요?", is_save=False)

        paint_history()

        query = st.chat_input("Type your message here...")
        if query:
            thread_id = get_thread_id()
            if not thread_id:
                thread_id = create_thread(query)

            send_message("user", query)
            with st.chat_message("assistant"):
                status_box = st.empty()
                status_ref = status_box.status(label="🐚 시작 중 ...", state="running")

                run = client.beta.threads.runs.create(
                    assistant_id=assistant_id,
                    thread_id=thread_id,
                )

                while True:
                    run = client.beta.threads.runs.retrieve(
                        thread_id=thread_id,
                        run_id=run.id,
                    )

                    if run.status == "in_progress":
                        status_ref.update(label="💡 생각 중 ...", state="running")

                    elif run.status == "requires_action":
                        status_ref.update(label="🔍 검색 중 ...", state="running")
                        tool_inputs = []

                        for tool_call in run.required_action.submit_tool_outputs.tool_calls:
                            function_info = functions_map.get(tool_call.function.name)
                            status_ref.update(label=function_info["status"], state="running")
                            tool_inputs.append({
                                "tool_call_id": tool_call.id,
                                "output": function_info["func"](json.loads(tool_call.function.arguments)),
                            })

                            run = client.beta.threads.runs.submit_tool_outputs(
                                thread_id=thread_id,
                                run_id=run.id,
                                tool_outputs=tool_inputs,
                            )

                    elif run.status == "completed":
                        status_ref.update(label="✅ 완료!", state="complete")
                        messages = client.beta.threads.messages.list(thread_id=thread_id)
                        for message in reversed(messages.data):
                            if message.role == "assistant":
                                message_text = message.content[0].text.value
                                st.markdown(message_text)
                                save_message("assistant", message_text)
                                break
                        status_box.empty()
                        st.rerun()
                        break

                    elif run.status in ["failed", "cancelled"]:
                        status_ref.update(label="❌ 실패!", state="error")
                        st.error(f"Run {run.status}.")
                        break
                    
                    time.sleep(1)


