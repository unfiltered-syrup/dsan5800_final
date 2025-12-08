import os
from datetime import datetime
from typing import Optional, Tuple, Dict, Any, List, Literal

from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_chroma import Chroma
from langchain_core.tools import tool
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from sqlalchemy import create_engine
import json


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

mpl.rcParams["font.family"] = "DejaVu Sans"
mpl.rcParams["figure.figsize"] = (10, 6)
mpl.rcParams["axes.titlesize"] = 16
mpl.rcParams["axes.labelsize"] = 13
mpl.rcParams["xtick.labelsize"] = 11
mpl.rcParams["ytick.labelsize"] = 11
mpl.rcParams["legend.fontsize"] = 11

COLOR_PRIMARY = "#1f77b4"
COLOR_SECONDARY = "#ff7f0e"
COLOR_TERTIARY = "#2ca02c"

ENGINE = create_engine("sqlite:///steam_clean_top2000.db")


def _style_axes(
    ax: plt.Axes, title: str, xlabel: Optional[str], ylabel: Optional[str]
) -> None:
    ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.3)


def _annotate_bar_values(ax: plt.Axes, bars) -> None:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.0f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            color="black",
        )


def _draw_single_plot(plot_cfg: Dict[str, Any], idx: int) -> None:
    chart_type = plot_cfg.get("chart_type")
    sql = plot_cfg.get("sql_query")
    x_field = plot_cfg.get("x_field")
    y_field = plot_cfg.get("y_field")
    reason = plot_cfg.get("reason", f"Plot {idx}")

    print(f"\n[Plot {idx}] {reason}")
    if not sql:
        print(f"[WARN] Plot {idx}: empty sql_query, skip.")
        return

    try:
        df = pd.read_sql_query(sql, ENGINE)
    except Exception as e:
        print(f"[ERROR] Plot {idx}: SQL failed.")
        print("Error:", e)
        print("SQL:", sql)
        return

    if df.empty:
        print(f"[INFO] Plot {idx}: query returned no rows, skip.")
        return

    print(f"[INFO] Plot {idx}: Data columns:", df.columns.tolist())

    fig, ax = plt.subplots()

    if chart_type == "bar":
        if x_field not in df.columns or y_field not in df.columns:
            print(f"[WARN] Plot {idx}: bar needs x_field={x_field}, y_field={y_field}.")
            plt.close(fig)
            return

        n = len(df)
        cmap = plt.cm.get_cmap("tab20", max(n, 3))
        colors = [cmap(i) for i in range(n)]

        bars = ax.bar(df[x_field], df[y_field], color=colors)
        plt.xticks(rotation=45, ha="right")
        _style_axes(ax, reason, x_field, y_field)

        from matplotlib.patches import Patch

        handles = [
            Patch(color=colors[i], label=str(df[x_field].iloc[i])) for i in range(n)
        ]
        ax.legend(
            handles=handles,
            title="Game",
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            borderaxespad=0.0,
        )
        _annotate_bar_values(ax, bars)

    elif chart_type == "line":
        if x_field not in df.columns or y_field not in df.columns:
            print(
                f"[WARN] Plot {idx}: line needs x_field={x_field}, y_field={y_field}."
            )
            plt.close(fig)
            return

        ax.plot(
            df[x_field],
            df[y_field],
            color=COLOR_PRIMARY,
            marker="o",
            label=y_field,
        )
        plt.xticks(rotation=45, ha="right")
        _style_axes(ax, reason, x_field, y_field)
        ax.legend()

    elif chart_type == "scatter":
        if x_field not in df.columns or y_field not in df.columns:
            print(
                f"[WARN] Plot {idx}: scatter needs x_field={x_field}, y_field={y_field}."
            )
            plt.close(fig)
            return

        ax.scatter(
            df[x_field],
            df[y_field],
            color=COLOR_PRIMARY,
            alpha=0.7,
            edgecolor="black",
            label=f"{y_field} vs {x_field}",
        )
        _style_axes(ax, reason, x_field, y_field)
        ax.legend()

    elif chart_type == "histogram":
        col = x_field or y_field
        if col not in df.columns:
            print(f"[WARN] Plot {idx}: histogram needs numeric column '{col}'.")
            plt.close(fig)
            return

        ax.hist(
            df[col].dropna(),
            bins=20,
            color=COLOR_PRIMARY,
            alpha=0.85,
            edgecolor="white",
            label=col,
        )
        _style_axes(ax, reason, col, "count")
        ax.legend()

    elif chart_type == "box":
        col = y_field or x_field
        if col not in df.columns:
            print(f"[WARN] Plot {idx}: box needs numeric column '{col}'.")
            plt.close(fig)
            return

        ax.boxplot(df[col].dropna(), patch_artist=True)
        for patch in ax.artists:
            patch.set_facecolor(COLOR_PRIMARY)
        _style_axes(ax, reason, "", col)

    elif chart_type == "pie":
        if x_field not in df.columns or y_field not in df.columns:
            print(f"[WARN] Plot {idx}: pie needs x_field={x_field}, y_field={y_field}.")
            plt.close(fig)
            return

        ax.pie(
            df[y_field],
            labels=df[x_field],
            autopct="%1.1f%%",
            startangle=140,
        )
        ax.set_title(reason)
        ax.axis("equal")

    else:
        print(f"[WARN] Plot {idx}: unknown chart_type '{chart_type}', skip.")
        plt.close(fig)
        return

    plt.tight_layout()
    plt.show()


def _visualize_from_plan_internal(plan: Dict[str, Any]) -> int:
    """Internal helper: draw all plots from a VIS_PLAN_JSON-like dict. Returns number of plots."""
    plots: List[Dict[str, Any]] = plan.get("plots") or []
    if not plots:
        print("[INFO] VIS_PLAN_JSON has no plots.")
        return 0

    count = 0
    for idx, p in enumerate(plots, start=1):
        _draw_single_plot(p, idx)
        count += 1
    return count


@tool
def visualize_from_plan(plan: Dict[str, Any]) -> str:
    """
    LangChain tool: given a VIS_PLAN_JSON-style dict with a top-level 'plots' list,
    execute the SQL queries, render matplotlib charts, and return a short summary.

    This tool is mainly for use by external callers or advanced agents; the CLI
    at the bottom of this file also calls the same internal logic.
    """
    if plan is None:
        return "No visualization plan was provided, so no plots were generated."
    try:
        n = _visualize_from_plan_internal(plan)
        if n == 0:
            return "No plots were generated because the plan contained an empty 'plots' list."
        return f"Rendered {n} plot(s) based on the visualization plan."
    except Exception as e:
        return f"Failed to render plots from plan due to error: {e}"


CUSTOM_TOOLS = [get_current_time, get_similar_games, visualize_from_plan]


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
            # f"You are a Steam game recommendation assistant. "
            # "You can use a SQLite database with tables 'steam_games' and 'steam_reviews' "
            # "for factual information such as tags, genres, prices, release dates, and "
            # "review sentiment. "
            # "You also have a tool called get_similar_games(category, query, k), which "
            # "finds games similar to a given input based on a chosen semantic category. "
            # "If get_similar_games return the same games in the same series, increase k and look for other similar games"
            # "The category option 'tags' focuses on user-defined tags, 'genres' focuses "
            # "on genre information, 'about' focuses on description and theme, 'reviews' "
            # "focuses on how players talk about the game, and 'mixed' combines all "
            # "available text fields into a single similarity profile. "
            # "Use SQL when the user needs exact filters or numeric comparisons. "
            # "Use get_similar_games when the user wants recommendations similar in style, "
            # "tone, mechanics, mood, or how players discuss the game. "
            # "When making recommendations, clearly explain why each game fits the request, "
            # "name the games directly, and include insights or sentiment summaries from "
            # "the 'steam_reviews' table when relevant. "
            # "Call get_current_time every time and be aware of the date of request"
            # "When recommending multiple games, always recommend the one with better reviews and more player counts"
            # "Don't recommend multiple games in the same franchise to user, but always recommand more than two games"
            "You are a Steam game recommendation assistant. "
            "You can use a SQLite database with tables 'steam_games_clean' and "
            "'steam_reviews_clean' for factual information such as tags, genres, "
            "prices, release dates, review counts, review sentiment, and playtime. "
            "You also have a tool called get_similar_games(category, query, k), which "
            "finds games similar to a given input based on a chosen semantic category. "
            "If get_similar_games returns multiple entries from the same game series, "
            "increase k and look for additional diverse games. "
            "The category option 'tags' focuses on user-defined tags, 'genres' focuses "
            "on genre information, 'about' focuses on description and theme, 'reviews' "
            "focuses on how players talk about the game, and 'mixed' combines all "
            "available text fields into a single similarity profile. "
            "Use SQL when the user needs exact filters or numeric comparisons. "
            "Use get_similar_games when the user wants recommendations similar in style, "
            "tone, mechanics, mood, or how players discuss the game. "
            "When making recommendations, clearly explain why each game fits the request, "
            "name the games directly, and include insights or sentiment summaries from "
            "the 'steam_reviews_clean' table when relevant. "
            "Always call get_current_time at least once per conversation to keep track "
            "of the current date. "
            "When recommending multiple games, prefer those with stronger reviews and "
            "more review counts, and avoid recommending multiple entries from the same "
            "franchise in a single answer. Always recommend more than two games when "
            "it makes sense for the user. "
            "In table 'steam_games_clean', numeric fields include rank (popularity rank, where smaller is more popular), recent_review_count, all_review_count, avg_playtime_forever, avg_playtime_review, positive_rate (fraction of positive reviews from 0 to 1), and useful_rate (fraction of helpful reviews). In table 'steam_reviews_clean', numeric fields include voted_up, votes_up, votes_funny, playtime_forever, playtime_at_review, and timestamps created_date and updated_date, which can be grouped by day, month, or year to show trends."

            "In addition to your normal answer, you MUST also decide whether one or more data visualizations would help the user. Choose visualization types based on the semantics of the question: if the user asks for the “fastest,” “highest,” “top,” “best,” or “most” game (for example: “the most popular game,” “the best,” “top game,” “highest rated game”), treat this as a ranking problem. In those cases, always consider plotting a bar chart comparing the top five or ten games on the relevant metric such as all_review_count, recent_review_count, positive_rate, or avg_playtime_forever, and also consider a scatter plot relating that metric to another numeric variable, such as review score versus price or playtime. When a question involves numeric metrics varying along another axis—such as review_score versus price, positive_rate versus all_review_count, or avg_playtime_forever versus positive_rate—you should prefer scatter plots, and optionally line plots if there is a clear ordering such as time or rank. When the question involves categories such as genres, tags, or developers, consider bar charts of aggregated values by category and boxplots showing the distribution of a metric within each category. Only draw a global histogram or distribution plot when the question explicitly asks about overall distributions—for example questions about “in general,” “on average,” or “how are games distributed”—and avoid drawing irrelevant global histograms that are not directly tied to the user’s intent. Whenever you use a numeric metric such as positive_rate, useful_rate, review counts, or playtime, you may add distributional visualizations such as histograms or boxplots to show spread and outliers, but they must always be clearly related to the user’s question. You may propose between zero and four plots in total, and you should prefer diversity in chart types rather than repeating similar ones; every plot you propose must feel targeted and analytical rather than generic."

            "For bar charts, each bar MUST have a distinct color determined by the game. Every bar chart MUST include a legend mapping each color to the game name, and each bar MUST display its numeric value above the bar. All charts must follow clear visual aesthetics with readable fonts, clean gridlines, proper spacing, and visually distinct colors."

            "When generating SQL for ranking queries such as top 5, best game, or most popular, you MUST NOT use LIMIT alone. Instead, you MUST perform tie-aware selection. If the N-th game shares the same metric value with additional games, you must include ALL games tied at that value so that the final set may exceed N. Preferred SQL patterns include using RANK() OVER (ORDER BY metric DESC) or selecting the threshold value with OFFSET and including all rows greater than or equal to that value."

            "At the end of your answer, you MUST output exactly one line beginning with “VIS_PLAN_JSON: ” followed by a single-line JSON object describing your visualization plan. This JSON object must have a top-level key named “plots,” whose value is a list of zero to four objects. Each object represents one plot and must contain the keys chart_type, reason, sql_query, x_field, and y_field. The chart_type must be one of “bar,” “line,” “scatter,” “histogram,” “box,” or “pie.” The reason must briefly explain what the plot shows and why it is relevant. The sql_query must be a valid SQL query over the tables 'steam_games_clean' or 'steam_reviews_clean' that returns the data needed for the plot. The x_field and y_field must be column names from the query result or null. For histograms, x_field should be the numeric column and y_field should be null. For boxplots, y_field should be the numeric column and x_field should be null. If no numeric visualization makes sense, the JSON must still be valid and must contain an empty list for “plots,” while your natural-language answer should explain why no plot is appropriate. The JSON must be syntactically valid: use double quotes for all keys and string values, never include trailing commas, and keep the entire JSON object on a single line after “VIS_PLAN_JSON: ” without adding any additional commentary."
        ),
        extra_tools=CUSTOM_TOOLS,
    )
    return agent


def split_answer_and_vis_plan(text: str) -> Tuple[str, Optional[Dict[str, Any]]]:
    lines = text.splitlines()
    vis_line = None
    for line in lines:
        if line.strip().startswith("VIS_PLAN_JSON:"):
            vis_line = line.strip()
            break

    if vis_line is None:
        return text, None

    main_lines = [l for l in lines if l.strip() != vis_line]
    main_text = "\n".join(main_lines).strip()

    json_str = vis_line.split("VIS_PLAN_JSON:", 1)[1].strip()
    try:
        plan = json.loads(json_str)
    except json.JSONDecodeError:
        print("[WARN] Failed to parse VIS_PLAN_JSON JSON. Raw line:")
        print(vis_line)
        return main_text, None

    return main_text, plan


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
        full_text = resp["output"]

        main_text, plan = split_answer_and_vis_plan(full_text)

        print("\nAgent:", resp["output"], "\n")
        if plan is not None:
            _visualize_from_plan_internal(plan)
        else:
            print("[INFO] No VIS_PLAN_JSON found; skipping visualization.")
        print('-*'*50)


if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("missing api key")
    agent = build_agent()
    interactive_loop(agent)
