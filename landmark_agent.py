import base64
import requests
import streamlit as st

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.globals import set_debug
from langchain.tools import Tool


st.set_page_config(
    page_title="Landmark Helper",
    page_icon="🧭",
    layout="centered"
)

with st.sidebar:
    st.title("Model Configuration")

    provider = st.radio(
        "Choose Provider",
        ["OpenAI", "Gemini"]
    )

    api_key = st.text_input(
        "API Key",
        type="password"
    )

if not api_key:
    st.info("Enter an API key to continue.")
    st.stop()
set_debug(False)

if provider == "OpenAI":
    vision_llm = ChatOpenAI(
        model="gpt-4o",
        api_key=api_key,
        temperature=0
    )

    llm = ChatOpenAI(
        model="gpt-4o",
        api_key=api_key,
        temperature=0
    )
else:
    vision_llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=api_key,
        temperature=0
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=api_key,
        temperature=0
    )


def wiki_search(query):
    headers = {"User-Agent": "LandmarkHelper/1.0 (student project)"}
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "format": "json",
    }

    response = requests.get(
        "https://en.wikipedia.org/w/api.php",
        params=params,
        headers=headers,
    )
    data = response.json()
    results = data["query"]["search"]

    if not results:
        return "No results found"

    return results[0]["title"]


wiki_tool = Tool(
    name="Wikipedia",
    func=wiki_search,
    description="Search Wikipedia for information.",
)


def encode_image(image_file):
    return base64.b64encode(image_file.read()).decode()


def build_vision_chain():
    vision_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant that can identify a landmark. Return only landmark name like Taj mahal ,trump tower without explanation",
            ),
            (
                "human",
                [
                    {"type": "text", "text": "return the landmark name"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/jpeg;base64,{image}",
                            "detail": "low",
                        },
                    },
                ],
            ),
        ]
    )
    return vision_prompt | vision_llm


def build_agent():
    ddg_tool = DuckDuckGoSearchRun()

    tools = [
        wiki_tool,
        ddg_tool
    ]

    prompt = hub.pull("hwchase17/react")

    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=prompt
    )

    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True
    )


def analyze_uploaded_landmark(uploaded_file, vision_chain):
    image_b64 = encode_image(uploaded_file)
    vision_response = vision_chain.invoke({"image": image_b64})
    return vision_response.content.strip().split("\n")[0]


def research_landmark(agent, landmark_name, question):
    task = (
        f"Landmark: {landmark_name} Question: {question} "
        "Use Wikipedia or DuckDuckGo if needed. Answer briefly and accurately."
    )
    response = agent.invoke({"input": task})
    return response["output"]


def main():
    st.title("Landmark Helper (Vision + ReAct Agent)")

    uploaded_file = st.file_uploader("Upload your image", type=["jpg", "png"])
    question = st.text_input("Enter a question about the landmark")

    if uploaded_file and question:
        vision_chain = build_vision_chain()
        agent = build_agent()
        with st.spinner("Identifying landmark"):
            landmark_name = analyze_uploaded_landmark(uploaded_file, vision_chain)

        st.success(landmark_name)

        with st.spinner("Researching"):
            answer = research_landmark(agent, landmark_name, question)

        st.write(answer)


if __name__ == "__main__":
    main()