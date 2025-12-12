# Project Report: A Hybrid RAG and SQL-Based Conversational Agent for Steam Game Discovery and Analytics

**Date:** December 2025

---
## 1. Introduction

### 1.1 Background and Motivation
With tens of thousands of games available on Steam, the volume of choices creates a discovery challenge for users. Traditional recommendation systems, which typically rely on user behavior patterns or basic tag matching, are effective for surfacing popular titles. However, these methods often lack the nuance to identify games based on complex narrative themes or specific stylistic requirements. Furthermore, they provide limited support for users who wish to make decisions based on quantitative data analysis. Specifically, existing systems struggle to address:

1.  **Semantic Discovery:** Finding games based on abstract concepts, "vibes," or narrative tones (e.g., "a game that feels like a rainy cyberpunk noir") rather than specific genre tags.
2.  **Analytical Decision Making:** Users often weigh purchasing decisions against objective metrics, such as price-to-playtime ratios, positive review rates, or historical trends, which usually requires manual data aggregation.

### 1.2 Project Objective
This project aims to build a **Conversational Game Assistant** that bridges the gap between qualitative recommendation and quantitative analysis. By integrating **Retrieval-Augmented Generation (RAG)** for semantic understanding and **SQL-based Tool Use** for factual precision, the agent serves as a comprehensive guide. A key differentiator of this system is its ability to autonomously generate **Data Visualizations** (charts and graphs) to support its textual arguments, providing users with a multi-modal interaction experience.

---

## 2. Related Work

### 2.1 LLMs in Recommendation Systems
Large Language Models (LLMs) have transformed recommendation systems by enabling conversational interfaces. Recent works have explored RAG to inject domain-specific knowledge into LLMs, reducing hallucinations. However, most existing LLM-based recommenders operate purely in the text domain.

### 2.2 Text-to-SQL and Data Visualization
The field of Text-to-SQL focuses on translating natural language questions into database queries. While agents like LangChain's SQLAgent exist, they rarely integrate deeply with visualization pipelines. This project extends standard SQL agents by introducing a **Visualization Protocol**, allowing the LLM to act not just as a database interface, but as a data analyst that plans and executes visual reporting.

---

## 3. Methods

The system employs a decoupled client-server architecture divided into **Data Engineering**, **Intelligent Backend**, and **Interactive Frontend** layers.

![System Architecture Diagram](./workflow.png)

### 3.1 Data Acquisition: Ethical Scraping Strategy
Direct usage of the official Steam API was deemed insufficient due to restrictive rate limits (approximately 1000 requests per user) and limited access to granular review data.
* **Implementation:** We utilized `BeautifulSoup` and `Requests` to perform ethical scraping of the Steam Store. The scraper runs on a schedule to fetch data for the top games.
* **Storage:** Raw HTML data is parsed and structured into a **SQLite** database (`steam_games` and `steam_reviews` tables).

### 3.2 Data Cleaning and Vectorization
Raw data undergoes a rigorous cleaning pipeline before ingestion:
1.  **Text Processing:** Removal of stopwords, lemmatization, and noise filtering (HTML tags, URLs).
2.  **Feature Extraction:** Calculation of `avg_playtime`, `positive_rate`, and `price_val` (numeric conversion).
3.  **Help Score Calculation:** Reviews are scored to prioritize quality using the formula:

$$
\text{help\_score} = 0.3 \times \log(1 + \text{votes\_up}) + 0.7 \times \text{info\_density}
$$

4.  **Vectorization:** Cleaned data is processed with `SentenceTransformer` ("all-MiniLM-L6-v2") and stored in a **Chroma Vector Store**. We created specific indices for **Tags**, **Genres**, **About**, **Reviews**, and a **Mixed** embedding field to support diverse similarity searches.

### 3.3 Agent Architecture and Toolset
The core reasoning engine is built on **LangChain** and **GPT-4o-mini**, hosted on a **FastAPI** backend. To solve complex user queries, the agent is equipped with a specialized set of custom tools:

* **`sql_db_query` (Left Brain):**
    * **Function:** Executes raw SQL queries against the `steam_clean.db` SQLite database.
    * **Use Case:** Handles factual questions requiring exact filtering, sorting, or aggregation (e.g., "What are the top 5 cheapest RPGs?", "Count games released in 2023").

* **`get_similar_games` (Right Brain):**
    * **Function:** Performs semantic similarity searches against ChromaDB using vector embeddings.
    * **Use Case:** Handles abstract requests about game "feel," narrative style, or thematic similarity (e.g., "Games like Stardew Valley but in space"). It supports searching specific metadata fields (Tags, Genres, About) or a "Mixed" profile.

* **`summarize_reviews` (Qualitative Analysis):**
    * **Function:** Retrieves the most helpful reviews (ranked by `help_score`) for a specific game ID and synthesizes them using the LLM.
    * **Use Case:** Provides concise bullet points on Pros/Cons, Performance issues, or Gameplay mechanics when a user asks "Is this game worth it?".

* **`visualize_from_plan` (Quantitative Analysis):**
    * **Function:** A structured protocol tool. Instead of executing Python directly, the agent generates a `VIS_PLAN_JSON` object containing chart types and data definitions.
    * **Use Case:** Triggered when the user's intent implies a visual comparison, trend, or distribution.

**Prompt Engineering (Few-Shot Strategy):**
A critical methodological contribution was the refinement of the System Prompt to guide the agent's tool selection. Initially, the agent biased heavily towards defaulting to bar charts for all visual queries. We implemented **Few-Shot Prompting**, providing the agent with "Thought-Plan" examples in the system message. These examples explicitly taught the agent to distinguish between analytic intents, ensuring it correctly selects Scatter Plots for correlations, Box Plots for distributions, and Line Charts for trends.

### 3.4 Visualization Strategy
To ensure consistent rendering, the system separates reasoning from visualization:
1.  **Intent Recognition:** The agent identifies if the user needs a **Ranking** (Bar), **Trend** (Line), **Distribution** (Histogram/Box), or **Correlation** (Scatter).
2.  **Data Retrieval:** The backend executes the SQL required for the plot.
3.  **Frontend Rendering:** The **Svelte** frontend receives the JSON plan and data, rendering interactive charts that adhere to consistent styling, legend, and color rules defined by the protocol.

---

## 4. Experiments and Validation

We evaluated the system using a set of real-world queries designed to stress-test the distinct subsystems: Vector Search, SQL Analytics, and Qualitative Summarization.

### 4.1 Scenario A: Semantic Discovery (Vector Search)
* **Query:** *"Recommend similar games to undertale"*
* **Mechanism:** The agent identified this as a request for "vibe" and narrative similarity rather than a strict genre lookup. It routed the request to the `get_similar_games` tool using the `mixed` vector collection.
* **Result:** Instead of simply returning generic RPGs, the system retrieved titles known for emotional storytelling, retro aesthetics, and player choices (e.g., *Omori*, *Deltarune*), validating the effectiveness of the embedding-based search over keyword matching.

### 4.2 Scenario B: Analytical Visualization (SQL + Plotting)
* **Query:** *"What is the popularity trend of the action genre? Visualize"*
* **Mechanism:** The agent parsed two distinct intents: "Trend" (Time-series data) and "Visualize" (Chart generation).
    1.  It executed a SQL query aggregating review counts or release frequency for games tagged "Action" over yearly intervals.
    2.  It generated a `VIS_PLAN_JSON` specifying a **Line Chart** (x-axis: Year, y-axis: Count).
* **Result:** The frontend rendered an interactive line chart showing the growth of the Action genre over the last decade, with the agent adding textual commentary on peak years.

### 4.3 Scenario C: Qualitative Review Synthesis (Summarization)
* **Query:** *"Review summary for infinity nikki"*
* **Mechanism:** The agent recognized a request for specific community feedback and invoked the `summarize_reviews` tool.
* **Action:** The system retrieved the top 80 review sentences for the specific AppID, ranked strictly by our `help_score` metric to filter out noise. The LLM then synthesized these into categories.
* **Result:** The agent outputted a structured summary with bullet points covering "Visuals/Art Style" (Pros) and "Gacha Mechanics/Performance" (Cons), successfully compressing hundreds of user reviews into a concise insight.

---

## 5. Results and Discussion

### 5.1 Hybrid Search Works Best
Our testing showed that neither SQL nor Vector search is enough on its own. If you ask for "Top 10 cheap games," Vector search fails. If you ask for "Games with a dark atmosphere," SQL fails. Combining them gave us the best of both worlds.

### 5.2 Visualization Matters
By forcing the agent to output a structured JSON plan for charts instead of trying to write Python code on the fly, we avoided a lot of errors. It also let us keep the frontend fast and interactive.

### 5.3 Deployment Status (Important Note)
We successfully deployed the full web application to a Google Cloud Platform (GCP) server at **`http://34.134.144.250/`**.

**Note on Availability:** If you cannot access the link right now, it is because **we have temporarily paused the server.** Since Google Cloud charges by the hour for active instances, keeping the server running 24/7 was running up a bill that exceeded our student budget. We turned off the resources after our final testing to save costs, but the deployment architecture is fully functional.

---

## 6. Limitations and Future Work

### 6.1 Limitations
* **Static Data:** Since we scrape data on a schedule, our database isn't real-time. If a game came out yesterday, our agent might not know about it yet.
* **Latency Trade-off:** The system averages 5-10 seconds per query. This latency is primarily driven by the **multi-step agentic workflow**, which involves sequential network round-trips: reasoning $\rightarrow$ tool execution (SQL/Vector) $\rightarrow$ data retrieval $\rightarrow$ final synthesis. While slower than a traditional keyword search, this trade-off is necessary to enable deep, multi-modal analysis and visualization.

### 6.2 Future Work
* **Multi-turn Context Awareness:** Currently, the agent treats each query independently. We plan to implement **conversational memory** (using LangChain's history modules) to support follow-up questions. For example, if a user asks for "RPGs," they should be able to simply ask "Which ones are under $10?" in the next turn without repeating the context.
* **Real-time Updates:** Integrating the official Steam API to fetch real-time price changes and player count updates to supplement the scraped static data.
* **User Feedback Loop:** Implementing a "thumbs up/down" mechanism on recommendations to fine-tune the retrieval relevance over time.

---

## 7. Conclusion
This project was a great learning experience in building a full-stack AI application. By combining rigorous data engineering (ETL) with a modern web stack (FastAPI + Svelte), we proved that LLMs can be more than just chatbots—they can be capable data analysts. While we had to be mindful of cloud costs for deployment, the final product successfully bridges the gap between finding a game and analyzing if it's worth buying.