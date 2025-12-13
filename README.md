# Steam Game Recommendation System

A conversational AI agent for Steam game discovery and analytics, combining semantic search (RAG) with structured SQL queries and automatic data visualization.

## Deployment

We successfully deployed the full web application to a Google Cloud Platform (GCP) server at **`http://34.134.144.250/`**.

**Note on Availability:** The server may be temporarily paused to manage costs. The deployment architecture is fully functional and can be replicated locally following the instructions below.

## Features

- **Hybrid Search**: Semantic similarity search via vector embeddings (tags, genres, reviews, descriptions) combined with precise SQL queries
- **Conversational Interface**: Natural language interaction powered by GPT-4o-mini via LangChain
- **Automatic Visualization**: Generates charts (bar, line, scatter, histogram, box, pie) based on user queries
- **Review Summarization**: Extracts and summarizes player feedback by category (gameplay, performance, content)
- **Web UI**: Svelte frontend with FastAPI backend

## Architecture

```
Frontend (Svelte) → FastAPI Backend → LangChain Agent
                                      ├── SQL Database (SQLite)
                                      ├── Vector Store (ChromaDB)
                                      └── Visualization Engine (Matplotlib)
```

## Tech Stack

- **Backend**: Python 3.13+, FastAPI, LangChain, OpenAI gpt-4.1-mini
- **Vector DB**: ChromaDB with Sentence Transformers (all-MiniLM-L6-v2)
- **Database**: SQLite
- **Frontend**: Svelte 5, Vite
- **Data Processing**: BeautifulSoup, Pandas, Matplotlib

## Quick Start

### Prerequisites

- Python 3.13+
- Node.js 16+
- OpenAI API key

### Installation

1. **Install dependencies**:
   ```bash
   uv sync  # or: pip install -r requirements.txt
   ```

2. **Set environment variable**:
   ```bash
   export OPENAI_API_KEY="your-api-key"
   ```

3. **Prepare data**:

   You have two options: use pre-generated database files (recommended) or generate them from scratch.

   **Option A: Use pre-generated data (recommended)**
   
   If you have access to the pre-generated database files, you can find them in the project repository or shared storage:
   - `steam_clean_top2000.db` - Cleaned SQLite database (required)
   - `steam_vector_store/` - ChromaDB vector store directory (required)

   Download the files from [Google Drive](https://drive.google.com/drive/folders/1GFd6v8lvrP3AeJF_rHX9AGC4QQ_BBxU4) and place **both files** in the **project root directory** (at the same level as `agent.py`). Once done, proceed to step 4.
   
   **Option B: Generate data from scratch**
   
   Follow these steps to generate the database files from Steam data:
   
   ```bash
   cd backend
   
   # Step 1: Scrape Steam game data
   python runner.py -u y -r y
   
   # Step 2: Clean and process the dataset
   python clean_dataset.py
   
   # Step 3: Build vector store
   python build_steam_vector_store.py
   
   # Step 4: Move files to project root (if needed)
   cd ..
   mv backend/steam_clean_top2000.db .
   mv backend/steam_vector_store .
   ```
   
   **Important Notes**:
   - The scraping process may take several hours depending on the number of games
   - Make sure you have sufficient disk space (database files can be several GB)
   - All final files (`steam_clean_top2000.db` and `steam_vector_store/`) must be in the project root directory

4. **Start the application**:

   The system consists of a backend API server and a frontend web interface. You need to run both:
   
   **Terminal 1 - Start Backend Server:**
   ```bash
   cd backend
   uvicorn main:app --reload --port 8000
   ```
   The backend API will be available at `http://localhost:8000`
   
   **Terminal 2 - Start Frontend Server:**
   ```bash
   cd steam-agent-ui
   npm install  # Only needed on first run
   npm run dev
   ```
   The frontend will be available at `http://localhost:5173`
   
   Open your browser and visit `http://localhost:5173` to use the web interface.

### CLI Usage

Run the agent directly:
```bash
python agent.py
```

## Project Structure

```
├── agent.py                    # Main agent logic (CLI version)
├── backend/
│   ├── main.py                 # FastAPI server
│   ├── steam_scraper.py        # Steam data scraper
│   ├── database_manager.py     # Database utilities
│   ├── clean_dataset.py        # Data cleaning
│   ├── build_steam_vector_store.py  # Vector store builder
│   └── runner.py               # Scraping runner
├── steam-agent-ui/             # Svelte frontend
└── pyproject.toml              # Python dependencies
```

## Core Components

### Agent Tools

- `get_similar_games(category, query, k)`: Vector similarity search across tags/genres/reviews/about/mixed
- `summarize_reviews(appid, max_sentences)`: LLM-powered review summarization
- `visualize_from_plan(plan)`: Executes visualization plans from JSON
- `get_current_time()`: Current date/time helper

### Vector Collections

- `steam_tags`: User-defined tags
- `steam_genres`: Game genres
- `steam_about`: Game descriptions
- `steam_reviews`: Review content
- `steam_mixed`: Combined text fields

## Example Queries

- "Recommend games similar to Undertale"
- "What are the top 10 cheapest RPGs?"
- "Show the popularity trend of action games" (with visualization)
- "Summarize reviews for [game name]"

## API

**POST** `/api/chat`

```json
{
  "message": "Recommend games like Stardew Valley"
}
```

Returns:
```json
{
  "answer": "Agent response text...",
  "raw_output": "Full output...",
  "plots": [{"chart_type": "bar", "reason": "...", "image_base64": "..."}]
}
```

## Data Files

The system requires two data files to run:

1. **`steam_clean_top2000.db`** - SQLite database containing cleaned game and review data
   - Location: Project root directory
   - Contains: `steam_games` and `steam_reviews` tables
   
2. **`steam_vector_store/`** - ChromaDB vector store directory
   - Location: Project root directory
   - Contains: Vector embeddings for semantic search (tags, genres, reviews, about, mixed)

Both files should be placed in the project root (same directory as `agent.py`).

## Notes

- Data is scraped periodically, not real-time
- Queries may take 5-10 seconds due to multi-step agent workflow
- **Required files**: `steam_clean_top2000.db` and `steam_vector_store/` directory must be in the project root
- Both backend and frontend servers must be running for the web interface to work
