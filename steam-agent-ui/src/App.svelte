<script>
  let userInput = "";
  let isLoading = false;

  let answer = "";
  let rawOutput = ""; 
  let plots = [];

  const API_URL = "http://localhost:8000/api/chat";

  async function sendMessage() {
    const msg = userInput.trim();
    if (!msg || isLoading) return;

    isLoading = true;
    answer = "";
    rawOutput = "";
    plots = [];

    try {
      const res = await fetch(API_URL, {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          message: msg
        })
      });

      if (!res.ok) {
        const text = await res.text();
        answer = `Backend error: ${res.status} ${text}`;
      } else {
        const data = await res.json();
        answer = data.answer || "";
        rawOutput = data.raw_output || "";
        plots = data.plots || [];
      }
    } catch (err) {
      console.error(err);
      answer = "Network error: " + err;
    } finally {
      isLoading = false;
    }
  }

  function handleKeydown(event) {
    if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
      sendMessage();
    }
  }
</script>

<style>
  :global(body) {
    margin: 0;
    font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    background: #f3f4f6;
    color: #111827;
  }

  .app {
    min-height: 100vh;
    display: flex;
    flex-direction: column;
  }

  .app-inner {
    max-width: 1200px;
    margin: 0 auto;
    padding: 1rem 1.25rem 1.5rem;
    box-sizing: border-box;
    width: 100%;
  }

  header {
    padding: 0.5rem 0 1.25rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.75rem;
  }

  .brand {
    display: flex;
    align-items: center;
    gap: 0.6rem;
  }

  .brand-mark {
    width: 28px;
    height: 28px;
    border-radius: 0.7rem;
    background: #111827;
    color: #f9fafb;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.9rem;
    font-weight: 600;
  }

  .brand-text h1 {
    margin: 0;
    font-size: 1.1rem;
    font-weight: 600;
  }

  .brand-text p {
    margin: 0.2rem 0 0;
    font-size: 0.8rem;
    color: #6b7280;
  }

  .status {
    font-size: 0.78rem;
    color: #6b7280;
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
  }

  .status-dot {
    width: 8px;
    height: 8px;
    border-radius: 999px;
    background: #10b981;
  }

  main {
    display: grid;
    grid-template-columns: minmax(0, 1.1fr) minmax(0, 1.4fr) minmax(0, 1.3fr);
    gap: 1rem;
  }

  @media (max-width: 1024px) {
    main {
      grid-template-columns: minmax(0, 1.1fr) minmax(0, 1.9fr);
      grid-template-areas:
        "input answer"
        "input plots";
    }

    .panel-input {
      grid-area: input;
    }

    .panel-answer {
      grid-area: answer;
    }

    .panel-plots {
      grid-area: plots;
    }
  }

  @media (max-width: 768px) {
    main {
      display: flex;
      flex-direction: column;
    }
  }

  .panel {
    background: #ffffff;
    border-radius: 0.75rem;
    border: 1px solid #e5e7eb;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }

  .panel-header {
    padding: 0.7rem 0.9rem;
    border-bottom: 1px solid #e5e7eb;
    font-size: 0.8rem;
    font-weight: 600;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }

  .panel-body {
    padding: 0.85rem 0.9rem 0.9rem;
    flex: 1;
    overflow: auto;
  }

  .input-group {
    display: flex;
    flex-direction: column;
    gap: 0.65rem;
  }

  textarea {
    width: 100%;
    min-height: 140px;
    resize: vertical;
    padding: 0.75rem 0.8rem;
    border-radius: 0.5rem;
    border: 1px solid #d1d5db;
    background: #f9fafb;
    color: #111827;
    font-size: 0.9rem;
    box-sizing: border-box;
    outline: none;
    transition: border-color 0.12s ease, background 0.12s ease;
  }

  textarea:focus {
    border-color: #2563eb;
    background: #ffffff;
  }

  .controls {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 0.75rem;
    flex-wrap: wrap;
  }

  button {
    border: none;
    border-radius: 999px;
    padding: 0.45rem 0.95rem;
    background: rgb(18, 29, 51);
    color: #f9fafb;
    font-size: 0.85rem;
    font-weight: 500;
    cursor: pointer;
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    transition: background 0.12s ease, transform 0.08s ease, opacity 0.1s ease;
  }

  button:hover:not(:disabled) {
    background: #030712;
  }

  button:active:not(:disabled) {
    transform: translateY(1px);
  }

  button:disabled {
    opacity: 0.6;
    cursor: default;
  }

  .hint {
    font-size: 0.78rem;
    color: #6b7280;
  }

  .hint code {
    font-size: 0.78rem;
    background: #f3f4f6;
    padding: 0.1rem 0.3rem;
    border-radius: 0.3rem;
  }

  pre {
    margin: 0;
    white-space: pre-wrap;
    word-break: break-word;
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
    font-size: 0.85rem;
    line-height: 1.5;
    color: #111827;
  }

  .loading {
    font-size: 0.85rem;
    color: #6b7280;
  }

  /* Plots */

  .plot-card {
    border-radius: 0.5rem;
    border: 1px solid #e5e7eb;
    padding: 0.6rem;
    margin-bottom: 0.75rem;
    background: #f9fafb;
  }

  .plot-type {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #9ca3af;
    margin-bottom: 0.25rem;
  }

  .plot-title {
    font-size: 0.82rem;
    font-weight: 500;
    margin-bottom: 0.45rem;
    color: #111827;
  }

  .plot-img {
    width: 100%;
    border-radius: 0.4rem;
    display: block;
  }

  @media (max-width: 640px) {
    header {
      flex-direction: column;
      align-items: flex-start;
      gap: 0.3rem;
    }
  }
</style>

<div class="app">
  <div class="app-inner">
    <header>
      <div class="brand">
        <div class="brand-mark">S</div>
        <div class="brand-text">
          <h1>Steam Guide</h1>
          <p></p>
        </div>
      </div>
      <div class="status">
        <span class="status-dot"></span>
        <span>{isLoading ? "Thinking…" : "Ready"}</span>
      </div>
    </header>

    <main>
      <section class="panel panel-input">
        <div class="panel-header">Prompt</div>
        <div class="panel-body">
          <div class="input-group">
            <textarea
              bind:value={userInput}
              placeholder="Ask for game recommendations, comparisons, review trends, or price vs rating analysis…"
              on:keydown={handleKeydown}
            ></textarea>

            <div class="controls">
              <button on:click={sendMessage} disabled={isLoading}>
                {#if isLoading}
                  Sending…
                {:else}
                  Send
                {/if}
              </button>
              <div class="hint">
                Press <code>Ctrl+Enter</code> (or <code>⌘+Enter</code>) to send
              </div>
            </div>
          </div>
        </div>
      </section>

      <section class="panel panel-answer">
        <div class="panel-header">Agent Response</div>
        <div class="panel-body">
          {#if isLoading}
            <div class="loading">
              Loading...
            </div>
          {/if}

          {#if answer}
            <pre>{answer}</pre>
          {/if}

          {#if !answer && !isLoading}
            <div class="hint">
              Try asking me a question!
            </div>
          {/if}
        </div>
      </section>

      <section class="panel panel-plots">
        <div class="panel-header">Visual Output</div>
        <div class="panel-body">
          {#if plots.length === 0}
            <div class="hint">
              Visuals will show up here.
            </div>
          {/if}

          {#each plots as p, i}
            <div class="plot-card">
              <div class="plot-type">
                {p.chart_type} · Plot {i + 1}
              </div>
              <div class="plot-title">{p.reason}</div>
              <img
                class="plot-img"
                alt={p.reason}
                src={`data:image/png;base64,${p.image_base64}`}
              />
            </div>
          {/each}
        </div>
      </section>
    </main>
  </div>
</div>