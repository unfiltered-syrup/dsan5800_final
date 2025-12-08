import sqlite3
import pandas as pd
import numpy as np
import re
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

RAW_DB = "steam_top_2000x100.db"
CLEAN_DB = "steam_clean_top2000.db"

print("Loading raw database:", RAW_DB)
conn_raw = sqlite3.connect(RAW_DB)

df_games = pd.read_sql("SELECT * FROM steam_games;", conn_raw)
df_reviews = pd.read_sql("SELECT * FROM steam_reviews;", conn_raw)
conn_raw.close()

nltk.download("punkt")
nltk.download("wordnet")
lemmatizer = WordNetLemmatizer()
tqdm.pandas()

stopwords_single = {
    "i","me","my","we","our","you","your","they","them","he","she","it","this","that","those","these",
    "a","an","the","of","to","in","on","for","with","about","from","and","or","but","so","because","as",
    "is","was","were","are","be","being","been","do","did","done","doing","have","has","had",
    "very","really","quite","just","literally","basically","actually","honestly","obviously",
    "yeah","yup","nope","fun","cool","awesome","amazing","fantastic","great","good","bad",
    "best","worst","better","worse","super","nice","enjoyable","experience","experiences",
    "review","reviews","recommend","recommended","story","plot","graphics","sound","music","vibe","nostalgia",
    "old","new","classic","gold","masterpiece","version","update","patch",
    "mod","mods","lol","lmao","xd","xdd","xddd","haha","hahaha","wtf","omg","btw","imo","idk",
    "bro","dude","man","guys","wow","damn","jeez","bruh","shit","fuck","fucking","hell","pls","plz","please",
    "nah","yup","yep","yeah","rip","ez","noob","pro",
    "juego","bueno","malo","mejor","peor","muy","bastante","este","esta","eso","olas","hola","gracias",
    "recomendo","recomendado","russian","rus","blyat","suka","debil","idiot","pizdec","govno","kak","eto","da","net",
    "game","games","gaming","gameplay","play","played","playing","player","players",
    "server","servers","community","dev","devs","valve","steam"
}   
stopwords_phrases = [
    "10/10","9/10","8/10","7/10","6/10",
    "10 10","9 10","8 10","7 10","6 10",
    "worth it","worth the money",
    "must play","must buy",
    "good game","great game","bad game",
    "still playable","still fun",
    "old but gold","classic game","old game"
]   

def clean_text(text):
    if not isinstance(text, str):
        return ""

    text = text.lower()

    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', " ", text)

    # Remove HTML
    text = re.sub(r"<.*?>", " ", text)

    # Keep basic chars
    text = re.sub(r"[^a-z0-9\s.,!?']", " ", text)

    # Remove multi-word stopwords
    for phrase in stopwords_phrases:
        text = re.sub(r"\b" + re.escape(phrase) + r"\b", " ", text)

    tokens = text.split()

    # Lemmatization
    tokens = [lemmatizer.lemmatize(t) for t in tokens]

    # Remove single-word stopwords
    tokens = [t for t in tokens if t not in stopwords_single]

    text = " ".join(tokens)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def is_noise(text):
    if len(text) < 10:
        return True
    non_alpha_ratio = len(re.sub("[a-zA-Z]", "", text)) / max(len(text), 1)
    if non_alpha_ratio > 0.6:
        return True
    if len(text.split()) < 3:
        return True
    return False


def normalize_date(x):
    x = str(x).strip()

    q = re.match(r"Q([1-4])\s+(\d{4})", x)
    if q:
        qnum, year = int(q.group(1)), int(q.group(2))
        month = (qnum - 1) * 3 + 1
        return pd.Timestamp(year, month, 1)

    h = re.match(r"H([1-2])\s+(\d{4})", x)
    if h:
        hnum, year = int(h.group(1)), int(h.group(2))
        month = 1 if hnum == 1 else 7
        return pd.Timestamp(year, month, 1)

    if x.lower() in {"tba", "tbd", "coming soon", "unknown"}:
        return pd.NaT

    return pd.to_datetime(x, errors="coerce", format="mixed")

df_games["clean_about"] = df_games["about"].fillna("").apply(clean_text)

# ==================================================
# 5. CLEAN REVIEWS
# ==================================================
df_reviews = df_reviews[df_reviews["language"] == "english"]
df_reviews["clean_review"] = df_reviews["review_text"].progress_apply(clean_text)

df_reviews_clean = df_reviews[~df_reviews["clean_review"].apply(is_noise)]
print("After noise removal:", df_reviews_clean.shape)

# ==================================================
# 6. PROCESS TIMESTAMPS
# ==================================================
df_games["release_date"] = df_games["release_date"].apply(normalize_date)
df_reviews_clean["created_date"] = pd.to_datetime(df_reviews_clean["timestamp_created"], unit="s")
df_reviews_clean["updated_date"] = pd.to_datetime(df_reviews_clean["timestamp_updated"], unit="s")
df_reviews_clean.drop(columns=["timestamp_created", "timestamp_updated"], inplace=True)

# ==================================================
# 7. PLAYTIME (>2h)
# ==================================================
df_filtered = df_reviews_clean[df_reviews_clean["playtime_forever"] >= 120]

df_play = (
    df_filtered.groupby("appid")
    .agg(
        avg_playtime_forever=("playtime_forever", "mean"),
        avg_playtime_review=("playtime_at_review", "mean"),
        positive_rate=("votes_up", lambda x: (x > 0).mean()),
    )
    .reset_index()
)

df_games = df_games.merge(df_play, on="appid", how="left")

# ==================================================
# 8. INFO_DENSITY via TF-IDF
# ==================================================
keywords = [
    # Gameplay
    "combat","melee","ranged","shooting","aim","accuracy","recoil",
    "movement","sprint","dash","dodge","roll","cover",
    "weapon","gun","rifle","pistol","shotgun","bow","sword",
    "reload","ammo","damage","hitbox",
    "enemy","boss","miniboss","npc","ai","behavior",
    "stealth","detection","sneak",
    "mechanic","system","feature","gameplay","loop",
    "mission","quest","objective","target","goal",
    "progression","build","perk","talent","skilltree","ability","cooldown",
    "craft","crafting","resource","blueprint",
    "loot","drop","inventory","equipment","armor","upgrade",

    # Performance
    "performance","optimization","lag","stutter","fps","framerate",
    "crash","freeze","bug","glitch","exploit",
    "loading","shader","compile","tearing","artifact",

    # Systems & AI
    "pathfinding","physics","collision","rng","spawn","scaling",
    "difficulty","balance","tuning",

    # Content / World
    "content","story","narrative","writing","plot","lore","worldbuilding",
    "level","map","zone","area","dungeon",
    "questline","campaign","chapter","sandbox","openworld","replayability",

    # Level Design
    "pacing","encounter","checkpoint","layout","architecture",

    # UI/UX & Controls
    "ui","hud","interface","accessibility","controls","feedback","animation",

    # Multiplayer
    "multiplayer","coop","co-op","server","ping","matchmaking","disconnect",
    "session","lobby","cheater","anticheat",

    # Economy/Monetization
    "economy","currency","reward","grind","paywall","microtransaction",
    "cosmetic","store","dlc","expansion",

    # Pricing / Value
    "price","value","worth","refund","expensive","overpriced","discount",

    # Audio/Visual
    "sound","audio","voice","mixing",
    "visual","graphics","lighting","effect"
]  

vectorizer = TfidfVectorizer(use_idf=False, binary=True)
tfidf_binary = vectorizer.fit_transform(df_reviews_clean["clean_review"])

vocab = vectorizer.vocabulary_
keyword_indices = []

for kw in keywords:
    for token, idx in vocab.items():
        if kw in token:
            keyword_indices.append(idx)

keyword_indices = sorted(set(keyword_indices))
df_reviews_clean["info_density"] = tfidf_binary[:, keyword_indices].sum(axis=1)

# ==================================================
# 9. HELP SCORE
# ==================================================
df_reviews_clean["help_score"] = (
    0.3 * np.log1p(df_reviews_clean["votes_up"]) +
    0.7 * df_reviews_clean["info_density"]
)

# ==================================================
# 10. USEFUL RATE PER GAME
# ==================================================
df_review_count = (
    df_reviews_clean.groupby("appid")
    .size()
    .reset_index(name="review_count")
)

df_useful = df_reviews_clean[df_reviews_clean["help_score"] > 0.2]

df_useful_count = (
    df_useful.groupby("appid")
    .size()
    .reset_index(name="useful_review_count")
)

df_join = df_review_count.merge(df_useful_count, on="appid", how="left")
df_join["useful_review_count"] = df_join["useful_review_count"].fillna(0)
df_join["useful_rate"] = df_join["useful_review_count"] / df_join["review_count"]

df_games = df_games.merge(df_join[["appid", "useful_rate"]], on="appid", how="left")

# ==================================================
# 11. SAVE CLEANED DATABASE
# ==================================================
print("Writing cleaned DB:", CLEAN_DB)
conn_clean = sqlite3.connect(CLEAN_DB)

df_games.to_sql("steam_games", conn_clean, if_exists="replace", index=False)
df_reviews_clean.to_sql("steam_reviews", conn_clean, if_exists="replace", index=False)

conn_clean.close()
print("Done! Saved to", CLEAN_DB)
