# build_steam_vector_store.py

import os
import sqlite3
from typing import Dict, List

from tqdm import tqdm

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma
from sentence_transformers import SentenceTransformer



DB_PATH = "steam_top_2000x100.db"

PERSIST_DIR = "steam_vector_store"

COLLECTIONS = {
    "tags": "steam_tags",
    "genres": "steam_genres",
    "about": "steam_about",
    "reviews": "steam_reviews",
    "mixed": "steam_mixed",
}



class LocalHFEmbeddings(Embeddings):

    def __init__(self, model_name = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        vecs = self.model.encode(texts, convert_to_numpy=True)
        return vecs.tolist()

    def embed_query(self, text):
        vec = self.model.encode([text], convert_to_numpy=True)[0]
        return vec.tolist()



def fetch_games(db_path = DB_PATH):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT
            appid,
            name,
            url,
            price,
            store_price,
            recent_review,
            overall_review,
            about,
            genres,
            user_tags,
            recent_review_count,
            all_review_count,
            developer
        FROM steam_games
        """
    )
    rows = cur.fetchall()
    conn.close()

    data: List[Dict] = []
    for row in rows:
        (
            appid,
            name,
            url,
            price,
            store_price,
            recent_review,
            overall_review,
            about,
            genres,
            user_tags,
            recent_review_count,
            all_review_count,
            developer,
        ) = row

        rec = {
            "appid": appid,
            "name": name or "",
            "url": url or "",
            "price": price or "",
            "store_price": store_price or "",
            "recent_review": recent_review or "",
            "overall_review": overall_review or "",
            "about": about or "",
            "genres": genres or "",
            "user_tags": user_tags or "",
            "recent_review_count": recent_review_count or 0,
            "all_review_count": all_review_count or 0,
            "developer": developer or "",
        }
        data.append(rec)
    return data



def build_vector_store():
    print("Fetching games from SQLite...")
    rows = fetch_games()
    print(f"Loaded {len(rows)} games.")

    embeddings = LocalHFEmbeddings(model_name="all-MiniLM-L6-v2")

    docs_by_category = {
        cat: [] for cat in COLLECTIONS.keys()
    }

    for rec in rows:
        base_meta = {
            "appid": rec["appid"],
            "name": rec["name"],
            "url": rec["url"],
            "price": rec["price"],
            "store_price": rec["store_price"],
            "recent_review": rec["recent_review"],
            "overall_review": rec["overall_review"],
            "about": rec["about"],
            "genres": rec["genres"],
            "user_tags": rec["user_tags"],
            "recent_review_count": rec["recent_review_count"],
            "all_review_count": rec["all_review_count"],
            "developer": rec["developer"],
        }

        # tags
        if rec["user_tags"].strip():
            docs_by_category["tags"].append(
                Document(
                    page_content=rec["user_tags"],
                    metadata={**base_meta, "field": "user_tags"},
                )
            )

        # genres
        if rec["genres"].strip():
            docs_by_category["genres"].append(
                Document(
                    page_content=rec["genres"],
                    metadata={**base_meta, "field": "genres"},
                )
            )

        # about
        if rec["about"].strip():
            docs_by_category["about"].append(
                Document(
                    page_content=rec["about"],
                    metadata={**base_meta, "field": "about"},
                )
            )

        # reviews (recent + overall)
        reviews_text = " ".join(
            [rec["recent_review"], rec["overall_review"]]
        ).strip()
        if reviews_text:
            docs_by_category["reviews"].append(
                Document(
                    page_content=reviews_text,
                    metadata={**base_meta, "field": "reviews"},
                )
            )

        # mixed (title + dev + genres + tags + reviews + about)
        mixed_text_parts = [
            f"Title: {rec['name']}",
            f"Developer: {rec['developer']}",
            f"Genres: {rec['genres']}",
            f"User tags: {rec['user_tags']}",
            f"Recent review summary: {rec['recent_review']}",
            f"Overall review summary: {rec['overall_review']}",
            f"About: {rec['about']}",
        ]
        mixed_text = "\n".join(mixed_text_parts)
        docs_by_category["mixed"].append(
            Document(
                page_content=mixed_text,
                metadata={**base_meta, "field": "mixed"},
            )
        )

    os.makedirs(PERSIST_DIR, exist_ok=True)

    for cat, collection_name in tqdm(COLLECTIONS.items(), desc="Building collections"):
        docs = docs_by_category[cat]
        if not docs:
            continue

        Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory=PERSIST_DIR,
            collection_name=collection_name,
        )



if __name__ == "__main__":
    build_vector_store()
