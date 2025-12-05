import os
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent


def build_steam_db() -> SQLDatabase:
    db_uri = "sqlite:///steam_top_5000.db"
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

    agent = create_sql_agent(
        llm=llm,
        db=db,
        agent_type="openai-tools",
        verbose=True,

        prefix=(
            "You are a Steam game recommendation assistant. "
            "You have access to a SQLite database with a table 'steam_games' "
            "containing information about the top 100 games. "
            "Use SQL to query that table when you need factual data. "
            "Then answer in natural language, citing game names and explaining your reasoning."
            "Always refer to comments and what people are saying about the game"
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
