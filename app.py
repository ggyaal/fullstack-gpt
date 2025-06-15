import streamlit as st
from openai import OpenAI, AssistantEventHandler
from langchain.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper
from langchain.utilities.wikipedia import WikipediaAPIWrapper
import requests
from bs4 import BeautifulSoup
import json
import os
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
    "search_duckduckgo": search_duckduckgo,
    "search_wikipedia": search_wikipedia,
    "website_scraping": website_scraping,
    "save_or_append_to_text": save_or_append_to_text,
}

class EventHandler(AssistantEventHandler):
    
    def __init__(self, status_ref, message_box=None, message=""):
        super().__init__()
        self.status_ref = status_ref
        self.message = message
        self.message_box = message_box or st.empty()

    def on_event(self, event):
        run_id = event.data.id

        if event.event == "thread.run.created":
            self.status_ref.update(label="📑 접수 중 ...", state="running")
        elif event.event == "thread.run.requires_action":
            self.status_ref.update(label="🔍 검색 중 ...", state="running")
            self.handle_required_action(event.data, run_id)
        elif event.event == "thread.message.created":
            self.status_ref.update(label="✍️ 응답 생성 중...", state="running")
            self.message_box = st.empty()
            self.message = ""
        elif event.event == "thread.run.completed":
            self.status_ref.update(label="✅ 완료!", state="complete")
            save_message("ai", self.message)
            st.rerun()

        elif event.event == "thread.run.failed":
            self.status_ref.update(label="❌ 실패!", state="error")
        return super().on_event(event)

    def handle_required_action(self, data, run_id):
        tool_outputs = []

        for tool in data.required_action.submit_tool_outputs.tool_calls:
            if tool.function.name == "search_duckduckgo":
                self.status_ref.update(label="🦆 DuckDuckGo 검색 중 ...", state="running")
            elif tool.function.name == "search_wikipedia":
                self.status_ref.update(label="📜 Wikipedia 검색 중 ...", state="running")
            elif tool.function.name == "website_scraping":
                self.status_ref.update(label="𝍌 웹사이트 스크래핑 중 ...", state="running")
            elif tool.function.name == "save_to_text":
                self.status_ref.update(label="💾 텍스트 저장 중 ...", state="running")
            tool_outputs.append({
                "tool_call_id": tool.id,
                "output": functions_map[tool.function.name](json.loads(tool.function.arguments)),
            })

        self.submit_tool_outputs(tool_outputs, run_id)

    def submit_tool_outputs(self, tool_outputs, run_id):
        handler = EventHandler(
            status_ref=self.status_ref,
            message_box=self.message_box,
            message=self.message,
        )

        with client.beta.threads.runs.submit_tool_outputs_stream(
            thread_id=self.current_run.thread_id,
            run_id=run_id,
            tool_outputs=tool_outputs,
            event_handler=handler,
        ) as stream:
            for text in stream.text_deltas:
                handler.message += text
                handler.message_box.markdown(handler.message)

client = None

def send_message(thread_id, query):
    return client.beta.threads.messages.create(
        thread_id=thread_id, role="user", content=query
    )

def create_thread(query):
    return client.beta.threads.create(
        messages=[
            {
                "role": "user",
                "content": query,
            }
        ]
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
        send_message("ai", "안녕하세요! 무엇을 검색할까요?", is_save=False)

        paint_history()

        query = st.chat_input("Type your message here...")
        if query:
            thread_id = get_thread_id()
            if not thread_id:
                thread_id = create_thread(query)

            send_message("user", query)
            with st.chat_message("ai"):
                with st.status(label="🐚 시작 중 ...", state="running") as status_ref:
                    with client.beta.threads.runs.stream(
                        thread_id=thread_id,
                        assistant_id=assistant_id,
                        event_handler=EventHandler(status_ref=status_ref),
                    ) as stream:
                        stream.until_done()

