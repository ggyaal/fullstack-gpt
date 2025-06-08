import os
import json
import streamlit as st
from langchain.retrievers import WikipediaRetriever

from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import StreamingStdOutCallbackHandler

st.set_page_config(
    page_title="QuizGPT",
    page_icon="⁉️",
)

def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)

def is_update(quiz_title, difficulty):
    pre_quiz_title = st.session_state["quiz_title"]
    pre_difficulty = st.session_state["difficulty"]

    u = pre_quiz_title != quiz_title or pre_difficulty != difficulty

    st.session_state["quiz_title"] = quiz_title
    st.session_state["difficulty"] = difficulty

    return u

@st.cache_data(show_spinner="Searching Wikipedia ..")
def search_wiki(topic):
    retriever = WikipediaRetriever(
        lang="ko",
        top_k_results=5,
    )
    return retriever.get_relevant_documents(topic)

@st.cache_data(show_spinner="Loading files...")
def split_file(file):
    file_content = file.read()
    file_path = f"./.cache/quiz_files/{file.name}"

    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, "wb") as f:
        f.write(file_content)
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path=file_path)
    docs = loader.load_and_split(text_splitter=splitter)

    return docs

@st.cache_data(show_spinner="Generating Quiz..")
def generate_quiz(_docs, topic, difficulty):
    func = {
        "name": "create_quiz",
        "description": "Function that takes a list of questions and answers and returns a quiz.",
        "parameters": {
            "type": "object",
            "properties": {
                "questions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "question": {"type": "string"},
                            "choices": {
                                "type": "array",
                                "items": {"type": "string"}
                            },
                            "answer_index": {"type": "integer"}
                        },
                        "required": ["question", "choices", "answer_index"]
                    }
                }
            },
            "required": ["questions"]
        }
    }

    llm = ChatOpenAI(
        temperature=0.1,
        model="gpt-4o-mini",
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()]
    ).bind(
        function_call={
            "name": "create_quiz",
        },
        functions=[
            func,
        ],
    )

    question_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are an AI that generates a list of 10 multiple-choice quiz questions in Korean, 
based on the given document and difficulty level.

Respond by **calling the `create_quiz` function** with one argument: `questions`.

Each question must have the following fields:
- `question`: a clearly phrased question in Korean
- `choices`: a list of 4 options in Korean (shuffled)
- `answer_index`: the 0-based index of the correct answer in the choices list

Guidelines:
- Do NOT explain anything.
- Only call the function and fill in the arguments.
- Make sure the correct answer index varies across the list.
- Ensure all choices are unique and relevant.
- Use casual Korean style but proper grammar.

Input Document:
{document}

Difficulty: {difficulty}
                """,
            )
        ]
    )

    question_chain = {
        "document": format_docs,
        "difficulty": lambda _: difficulty,
    } | question_prompt | llm

    response = question_chain.invoke(_docs)
    return json.loads(response.additional_kwargs.get("function_call")["arguments"])

with st.sidebar:
    docs = None
    topic = None
    difficulty = "easy"

    api_key = st.text_input(
        label="OPEN AI API KEY",
        placeholder="enter your OPEN AI API KEY!"
    )

    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

        choice = st.selectbox(
            "Choose what you want to use.",
            (
                "File",
                "Wikipedia Article",
            ),
        )

        if choice == "File":
            file = st.file_uploader(
                "Upload a .docx , .txt or .pdf file",
                type=["pdf", "txt", "docx"],
            )
            if file:
                docs = split_file(file)
        else:
            topic = st.text_input("Search Wikipedia...")
            if topic:
                docs = search_wiki(topic)

st.title("🧠 QuizGPT")


if not docs:
    st.markdown(
        """
**QuizGPT는 문서에서 자동으로 퀴즈를 생성해주는 AI 도구입니다.**

텍스트를 입력하기만 하면, QuizGPT가 핵심 내용을 분석하고 **4지선다형 객관식 퀴즈 10문제**를 자동으로 만들어줍니다.  
공부한 내용을 복습하거나, 시험 전 자가진단을 할 때 유용합니다.

---

## ✨ 주요 기능

- 📄 문서를 기반으로 퀴즈 자동 생성
- ❓ 4지선다형 객관식 문제 10개 생성
- 🇰🇷 한국어 질문 생성 지원
- 🎯 난이도를 넣어봤는데 차이는 별로 없는 것 같네요..

---
    """
    )
    st.session_state["is_submit"] = False
    st.session_state["quizs"] = None
    st.session_state["quiz_title"] = None
    st.session_state["difficulty"] = difficulty
    st.session_state["results"] = {}
else:
    quiz_title = topic if topic else file.name

    with st.sidebar:
        difficulty = st.selectbox("난이도", ["easy", "medium", "hard"])
        start = st.button("Generate Quiz")

        if start and is_update(quiz_title, difficulty):
            st.session_state["quizs"] = generate_quiz(docs, quiz_title, difficulty)

    if not st.session_state["quizs"]:
        st.markdown("""
### 사이드바의 생성 버튼을 눌러주세요! 🎰
        """)
    else:
        result = st.session_state["results"].get((st.session_state["quiz_title"], st.session_state["difficulty"]))

        if not result:
            answers = {i: None for i in range(10)}
            with st.form("question_quiz"):
                st.markdown(f"""
### {st.session_state["quiz_title"]}
**난이도 : {st.session_state["difficulty"]}**

----
                """)
                for idx, question in enumerate(st.session_state["quizs"]["questions"]):
                    value = st.radio(
                        f"Q{idx + 1}. {question['question']}",
                        [answer for answer in question["choices"]],
                        index=None
                    )
                    if value:
                        selected = question["choices"].index(value)
                        answers[idx] = selected

                button = st.form_submit_button()

                if button:
                    st.session_state["results"][(st.session_state["quiz_title"], st.session_state["difficulty"])] = {
                        "submitted": True,
                        "answers": answers,
                        "isEnd": False
                    }
                    st.session_state["is_submit"] = True
                    st.rerun()

        else:
            score = 0
            for idx, question in enumerate(st.session_state["quizs"]["questions"]):
                st.markdown(f"#### Q{idx + 1}. {question['question']}")
                answer_index = result.get("answers")[idx]

                if answer_index is not None:
                    answer = question["choices"][answer_index]

                    if answer_index == question["answer_index"]:
                        st.success(f"{answer} (정답!)")
                        score += 1
                    else:
                        st.error(f"{answer} (오답..)")
                else:
                    st.error("선택하지 않으셨습니다..")

                retry_button, end_button = st.columns([2, 1])

            if not score == 10:
                if not result.get("isEnd"):
                    with retry_button:
                        if st.button("다시 풀기", use_container_width=True):
                            st.session_state["results"].pop((st.session_state["quiz_title"], st.session_state["difficulty"]), None)
                            st.rerun()

                    with end_button:
                        if st.button("끝내기", use_container_width=True):
                            st.session_state["results"][(st.session_state["quiz_title"], st.session_state["difficulty"])]["isEnd"] = True
                            st.rerun()
                else:
                    st.metric("최종 점수", f"{score} / 10")
                    st.progress(score * 10)
            else:
                st.session_state["results"][(st.session_state["quiz_title"], st.session_state["difficulty"])]["isEnd"] = True
                st.metric("최종 점수", f"{score} / 10")
                st.progress(score * 10)

                if st.session_state["is_submit"]:
                    st.balloons()
                    st.session_state["is_submit"] = False