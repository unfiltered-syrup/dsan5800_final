import os
from datetime import datetime
from typing import Any, Dict, List, Literal

from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_chroma import Chroma
from langchain_core.tools import tool
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer


PERSIST_DIR = "steam_vector_store"

COLLECTIONS = {
    "tags": "steam_tags",
    "genres": "steam_genres",
    "about": "steam_about",
    "reviews": "steam_reviews",
    "mixed": "steam_mixed",
}



class LocalHFEmbeddings(Embeddings):
    """
    Local embedding model using sentence-transformers.

    Make sure the model_name matches what you used when building the
    vector store, or delete the old Chroma folder and rebuild if you change it.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        vecs = self.model.encode(texts, convert_to_numpy=True)
        return vecs.tolist()

    def embed_query(self, text: str) -> List[float]:
        vec = self.model.encode([text], convert_to_numpy=True)[0]
        return vec.tolist()


def build_steam_db() -> SQLDatabase:
    db_uri = "sqlite:///steam_top_2000x100.db"
    db = SQLDatabase.from_uri(db_uri)
    return db


def build_llm():
    return ChatOpenAI(
        model="gpt-4.1-mini",
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY"),
    )



def load_vector_stores() -> Dict[str, Chroma]:
    embeddings = LocalHFEmbeddings()
    vs: Dict[str, Chroma] = {}

    for cat, collection_name in COLLECTIONS.items():
        vs[cat] = Chroma(
            persist_directory=PERSIST_DIR,
            collection_name=collection_name,
            embedding_function=embeddings,
        )
    return vs


VECTOR_STORES = load_vector_stores()


@tool
def get_current_time(_: str = "") -> str:
    """Use this to find out the current date and time."""
    from datetime import datetime as _dt
    return _dt.now().strftime("%Y-%m-%d %H:%M:%S")


@tool
def get_similar_games(
    category: Literal["tags", "genres", "about", "reviews", "mixed"],
    query: str,
    k: int = 20,
) -> List[Dict[str, Any]]:
    """
    Find games similar to a given input, restricting similarity to a chosen category.

    Args:
        category:
            "tags": similarity based ONLY on user_tags
            "genres": similarity based ONLY on genres
            "about": similarity based ONLY on the 'about this game' text
            "reviews": similarity based ONLY on review summaries
            "mixed": similarity based on a combined profile of title, tags,
                     genres, review summaries, and about text
        query:
            Can be a game name, a list of tags/genres, or a free-text description.
        k:
            Maximum number of similar games to return.

    Returns:
        A list of dicts with game metadata that you can use in your response.
    """
    store = VECTOR_STORES.get(category)
    if store is None:
        store = VECTOR_STORES["mixed"]

    docs = store.similarity_search(query, k=k)

    results: List[Dict[str, Any]] = []
    for d in docs:
        m = d.metadata
        results.append(
            {
                "appid": m.get("appid"),
                "name": m.get("name"),
                "url": m.get("url"),
                "developer": m.get("developer"),
                "store_price": m.get("store_price") or m.get("price"),
                "recent_review": m.get("recent_review"),
                "overall_review": m.get("overall_review"),
                "recent_review_count": m.get("recent_review_count"),
                "all_review_count": m.get("all_review_count"),
                "genres": m.get("genres"),
                "user_tags": m.get("user_tags"),
                "category_used": category,
            }
        )
    return results


CUSTOM_TOOLS = [get_current_time, get_similar_games]



def build_agent():
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("missing api key")

    llm = build_llm()
    db = build_steam_db()
    date = datetime.now().strftime("%m/%d/%Y")

    agent = create_sql_agent(
        llm=llm,
        db=db,
        agent_type="openai-tools",
        verbose=True,
        prefix=(
            f"You are a Steam game recommendation assistant. "
            "You can use a SQLite database with tables 'steam_games' and 'steam_reviews' "
            "for factual information such as tags, genres, prices, release dates, and "
            "review sentiment. "
            "You also have a tool called get_similar_games(category, query, k), which "
            "finds games similar to a given input based on a chosen semantic category. "
            "If get_similar_games return the same games in the same series, increase k and look for other similar games"
            "The category option 'tags' focuses on user-defined tags, 'genres' focuses "
            "on genre information, 'about' focuses on description and theme, 'reviews' "
            "focuses on how players talk about the game, and 'mixed' combines all "
            "available text fields into a single similarity profile. "
            "Use SQL when the user needs exact filters or numeric comparisons. "
            "Use get_similar_games when the user wants recommendations similar in style, "
            "tone, mechanics, mood, or how players discuss the game. "
            "When making recommendations, clearly explain why each game fits the request, "
            "name the games directly, and include insights or sentiment summaries from "
            "the 'steam_reviews' table when relevant. "
            "Call get_current_time every time and be aware of the date of request"
            "When recommending multiple games, always recommend the one with better reviews and more player counts"
            "Don't recommend multiple games in the same franchise to user, but always recommand more than two games"
        ),
        extra_tools=CUSTOM_TOOLS,
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
        print('-*'*50)


if __name__ == "__main__":
    agent = build_agent()
    interactive_loop(agent)
