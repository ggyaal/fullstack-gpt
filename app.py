import streamlit as st
import json

from langchain.document_loaders import SitemapLoader
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda

st.set_page_config(
    page_title="SiteGPT",
    page_icon="📑",
)

llm = None
embedding = None

func = {
    "name": "create_message",
    "description": "Function that takes a comment, source with date and returns a message",
    "parameters": {
        "type": "object",
        "properties": {
            "comment": {
                "type": "string",
                "description": "The comment content"
            },
            "source": {
                "type": "string",
                "description": "The source URL or reference"
            },
            "date": {
                "type": "string",
                "description": "The date of the source, in YYYY-MM-DD format"
            }
        },
        "required": ["comment", "source", "date"]
    }
}

def get_answers(inputs):
    question = inputs["question"]
    contexts = inputs["contexts"]

    answers_prompt = ChatPromptTemplate.from_template(
        """
        Using ONLY the following context answer the user's question. If you can't just say you don't know, don't make anything up.

        Then, give a score to the answer between 0 and 5.

        If the answer answers the user question the score should be high, else it should be low.

        Make sure to always include the answer's score even if it's 0.

        Answer in the language of the user's question.

        Context: {context}

        Examples:

        Question: How far away is the moon?
        Answer: The moon is 384,400 km away.
        Score: 5

        Question: How far away is the sun?
        Answer: I don't know
        Score: 0

        Your turn!

        Question: {question}
    """
    )

    answers_chain = answers_prompt | llm

    return {
        "question": question,
        "answers": [
            {
                "answer": answers_chain.invoke(
                    {"question": question, "context": context.page_content}
                ).content,
                "source": context.metadata["source"],
                "date": context.metadata["lastmod"],
            }
            for context in contexts
        ],
    }

def choose_answer(inputs):
    question = inputs["question"]
    answers = inputs["answers"]

    choose_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                Use ONLY the following pre-existing answers to answer the user's question.

                Use the answers that have the highest score (more helpful) and favor the most recent ones.

                Cite sources and return the sources of the answers as they are, do not change them.

                Answers: {answers}
                """,
            ),
            ("human", "{question}"),
        ]
    )

    choose_llm = llm.bind(
        function_call={
            "name": "create_message"
        },
        functions=[
            func,
        ]
    )

    choose_chain = choose_prompt | choose_llm

    condensed = "\n\n".join(
        f"{answer['answer']}\nSource:{answer['source']}\nDate:{answer['date']}\n"
        for answer in answers
    )
    return choose_chain.invoke(
        {
            "question": question,
            "answers": condensed,
        }
    )

def parse_page(soup):
    header = soup.find("header")
    footer = soup.find("footer")

    if header:
        header.decompose()
    if footer:
        footer.decompose()
    
    return (
        str(soup.get_text())
        .replace("\n", " ")
        .replace("\xa0", " ")
    )

@st.cache_data(show_spinner="Loading websites...")
def load_website(url):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=600,
        chunk_overlap=100,
    )

    loader = SitemapLoader(
        url,
        filter_urls=[
            r"^(.*\/ai-gateway\/).*",
            r"^(.*\/vectorize\/).*",
            r"^(.*\/workers-ai\/).*",
        ],
        parsing_function=parse_page
    )
    loader.requests_per_second = 1
    docs = loader.load_and_split(text_splitter=splitter)
    vector_store = FAISS.from_documents(docs, embedding)
    
    return vector_store.as_retriever()

def save_message(message, role, source=None, date=None):
    st.session_state["messages"].append({"message": message, "role": role, "source": source, "date": date})
    with st.sidebar:
        st.write(st.session_state["messages"])

def send_message(message, role, source=None, date=None, save=True):
    with st.chat_message(role):
        st.markdown(message)
        if source and date:
            st.info(f"""
                출처: **{source}**
                _{date}_
            """)

    if save:
        save_message(message, role, source, date)

def paint_history():
    for message in st.session_state["messages"]:
        send_message(
            message=message["message"],
            role=message["role"],
            source=message["source"],
            date=message["date"],
            save=False,
        )

@st.cache_data(show_spinner="thinking ..")
def ai_answer(query):
    chain = (
        {
            "question": RunnablePassthrough(),
            "contexts": retriever,
        }
        | RunnableLambda(get_answers)
        | RunnableLambda(choose_answer)
    )
    result = chain.invoke(query)
    return json.loads(result.additional_kwargs.get("function_call")["arguments"])

st.title("SiteGPT")

with st.sidebar:
    api_key = st.text_input(
        label="OPEN AI API KEY",
        placeholder="enter your OPEN AI API KEY!"
    )
    st.markdown("----")
    st.link_button("🔗 Github Repository", url="https://github.com/ggyaal/fullstack-gpt/tree/siteGPT")

if api_key:
    llm = ChatOpenAI(
        temperature=0.1,
        model="gpt-4o-mini",
        openai_api_key=api_key,
    )

    embedding = OpenAIEmbeddings(
        openai_api_key=api_key
    )

    retriever = load_website("https://developers.cloudflare.com/sitemap-0.xml")

    paint_history()
    send_message("오케이! 질문해줘:)", "ai", save=False)
    query = st.chat_input("'Cloudflare'에 대해 질문하세요.")
    if query:
        send_message(query, "human")
        message = ai_answer(query)

        send_message(message["comment"], role="ai", source=message["source"], date=message["date"])

else:
    st.session_state["messages"] = []
    st.markdown("""
**siteGPT는 Cloudflare 사이트의 정보를 바탕으로 질문에 대한 답을 해주는 AI 챗봇입니다.**

---
    """)