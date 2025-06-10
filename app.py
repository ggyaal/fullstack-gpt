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

llm = ChatOpenAI(
    temperature=0.1,
    model="gpt-4o-mini",
    openai_api_key="OPENAI_API_KEY",
)

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
    vector_store = FAISS.from_documents(docs, OpenAIEmbeddings())
    
    return vector_store.as_retriever()

@st.cache_data(show_spinner="thinking ..")
def save_message(query):
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
    llm.openai_api_key = api_key

    retriever = load_website("https://developers.cloudflare.com/sitemap-0.xml")

    query = st.text_input("Ask a question to the website.")
    if query:
        message = save_message(query)

        with st.chat_message("ai"):
            st.markdown(f"**{message['comment']}**")
            st.info(f"""
                출처: **{message['source']}**
                _{message['date']}_
            """)
