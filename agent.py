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
    db_uri = "sqlite:///steam_clean_top2000.db"
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
            "You are a Steam game recommendation assistant. "
            "You can use a SQLite database with tables 'steam_games_clean' and "
            "'steam_reviews_clean' for factual information. "
            "In 'steam_games_clean', key numeric fields are: price_val (numeric price), "
            "positive_rate (0.0-1.0), all_review_count, recent_review_count, "
            "avg_playtime_forever, and rank. "
            "You also have a tool called get_similar_games(category, query, k) for semantic search. "
            
            "STRATEGY: "
            "1. Use SQL tools when the user asks for exact stats, rankings, counts, or numeric filtering. "
            "2. Use get_similar_games when the user asks for style, vibe, plot, or 'games like X'. "
            "3. If get_similar_games returns multiple entries from the same series, look for more diverse options. "
            "4. Always call get_current_time to know the date. "
            
            "VISUALIZATION RULES (CRITICAL): "
            "In addition to your text answer, you MUST decide if a visualization helps using the following STRICT rules. "
            "Do NOT default to bar charts. Analyze the user's analytic intent: "
            
            "1. USE 'SCATTER' CHART: When the user asks about the RELATIONSHIP, CORRELATION, or TRADE-OFF between two metrics. "
            "   - Example: 'Is expensive game better?' -> x_field='price_val', y_field='positive_rate'. "
            "   - Example: 'Price vs Playtime' -> x_field='price_val', y_field='avg_playtime_forever'. "
            "   - NOTE: Always use 'price_val' for plotting price, NEVER 'price'. "
            
            "2. USE 'BOX' CHART: When the user asks about DISTRIBUTIONS, SPREAD, or VARIABILITY within categories. "
            "   - Example: 'How much do RPGs usually cost?' -> y_field='price_val'. "
            "   - Example: 'Distribution of playtime for Action games'. "
            
            "3. USE 'BAR' CHART: ONLY when comparing specific values across specific named games or categories. "
            "   - Example: 'Top 5 games by reviews', 'Compare ratings of Game A and Game B'. "
            "   - Requirement: Each bar MUST have a distinct color mapped to the game name. "
            
            "4. USE 'PIE' CHART: When the user asks about PROPORTIONS or COMPOSITION. "
            "   - Example: 'Percentage of Free vs Paid games', 'Genre market share'. "
            
            "5. USE 'LINE' CHART: When the user asks about TRENDS over time. "
            "   - Example: 'Review trends over years'. "
            
            "6. USE 'HISTOGRAM': When asking for the general distribution of a single metric across the whole database. "
            "   - Example: 'Distribution of review scores'. "

            "SQL GENERATION RULES: "
            "When generating SQL for ranking (e.g., top 5), do NOT use LIMIT alone. "
            "Use tie-aware logic (e.g., WHERE value >= (SELECT value FROM ... LIMIT 1 OFFSET 4)). "
            
            "OUTPUT FORMAT: "
            "At the end of your answer, you MUST output exactly one line beginning with 'VIS_PLAN_JSON: ' followed by a single-line JSON object. "
            "Keys: 'plots' (list of objects). Each object has: 'chart_type' (bar, line, scatter, histogram, box, pie), 'reason', 'sql_query', 'x_field', 'y_field'. "
            "The 'sql_query' MUST use correct column names from 'steam_games_clean' (e.g., price_val, positive_rate). "
            "If no plot is needed, 'plots' should be an empty list. "
            "Ensure the JSON is valid, single-line, and has no trailing text."
        ),
        extra_tools=CUSTOM_TOOLS,
    ),
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