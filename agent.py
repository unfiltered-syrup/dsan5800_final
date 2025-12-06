import os
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from datetime import datetime


def build_steam_db() -> SQLDatabase:
    db_uri = "sqlite:///steam_top_100x200.db"
    db = SQLDatabase.from_uri(db_uri)
    return db


def build_llm():
    return ChatOpenAI(
        model="gpt-4.1-mini",
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY")
    )


def build_agent():
    llm = build_llm()
    db = build_steam_db()
    date = datetime.now().strftime("%m/%d/%Y")

    agent = create_sql_agent(
        llm=llm,
        db=db,
        agent_type="openai-tools",
        verbose=True,

        prefix=(
            f"You are a Steam game recommendation assistant. Today's date is {date}, "
            "You have access to a SQLite database with a table 'steam_games' and 'steam_reviews'"
            "containing information about the top games on steam. "
            "Use SQL to query that table when you need factual data. "
            "Then answer in natural language, citing game names and explaining your reasoning."
            "Always give examples of comments from players, and summarize the overall sentiment"
        ),
    )
    return agent



def interactive_loop(agent):

    while True:
        try:
            user_input = input("You: ")
        except (EOFError, KeyboardInterrupt):
            break

        if user_input.strip().lower() in {"exit", "quit"}:
            break

        if not user_input.strip():
            continue

        resp = agent.invoke({"input": user_input})
        print("\nAgent:", resp["output"], "\n")


if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("missing api key")

    agent = build_agent()

    interactive_loop(agent)
