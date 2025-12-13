import io
import base64
from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from sqlalchemy import create_engine

from agent import build_agent, split_answer_and_vis_plan



app = FastAPI()
ENGINE = create_engine("sqlite:///steam_clean_top2000.db")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Build the agent once and reuse it
AGENT = build_agent()


# Request/response models

class ChatRequest(BaseModel):
    message: str


class PlotImage(BaseModel):
    chart_type: str
    reason: str
    image_base64: str


class ChatResponse(BaseModel):
    answer: str
    raw_output: str
    plots: List[Dict[str, Any]]


# Helper functions and constants

MAX_LABEL_LEN = 18


def _truncate_label(s: Any, max_len: int = MAX_LABEL_LEN) -> str:
    """Convert to string and truncate if too long."""
    s = "" if s is None else str(s)
    if len(s) <= max_len:
        return s
    return s[: max_len - 1] + "…"


def _clean_xy(df: pd.DataFrame, x_field: str, y_field: str) -> pd.DataFrame:
    """
    Drop rows where x or y is null, and coerce x to string if it's object
    """
    if x_field not in df.columns or y_field not in df.columns:
        return pd.DataFrame(columns=[x_field, y_field])

    df_clean = df[[x_field, y_field]].dropna().copy()

    if df_clean[x_field].dtype == "object":
        df_clean[x_field] = df_clean[x_field].astype(str)

    return df_clean


def render_plot_to_base64(plot_cfg: Dict[str, Any], idx: int) -> Optional[Dict[str, Any]]:
    chart_type = plot_cfg.get("chart_type")
    sql = plot_cfg.get("sql_query")
    x_field = plot_cfg.get("x_field")
    y_field = plot_cfg.get("y_field")
    reason = plot_cfg.get("reason", f"Plot {idx}")

    print(f"\n[Plot {idx}] {reason}")
    if not sql:
        print(f" Plot {idx}: sql_query is empty, skipping.")
        return None

    try:
        df = pd.read_sql_query(sql, ENGINE)
    except Exception as e:
        print(f"[ERROR] Plot {idx}: failed to run SQL.")
        print("Error:", e)
        print("SQL:", sql)
        return None

    if df.empty:
        print(f" Plot {idx}: SQL returned no rows, skipping.")
        print("SQL:", sql)
        return None

    print(f" Plot {idx}: Data columns:", df.columns.tolist())

    mpl.rcParams["font.family"] = "DejaVu Sans"
    mpl.rcParams["figure.figsize"] = (10, 6)
    mpl.rcParams["axes.titlesize"] = 16
    mpl.rcParams["axes.labelsize"] = 13
    mpl.rcParams["xtick.labelsize"] = 11
    mpl.rcParams["ytick.labelsize"] = 11
    mpl.rcParams["legend.fontsize"] = 11

    COLOR_PRIMARY = "#1f77b4"

    fig, ax = plt.subplots()

    def _style_axes(ax, title, xlabel, ylabel):
        ax.set_title(title)
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", alpha=0.3)

    def _annotate_bar_values(ax, bars):
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

    # BAR
    if chart_type == "bar":
        if x_field not in df.columns or y_field not in df.columns:
            print(f" Plot {idx}: bar chart needs x_field={x_field} and y_field={y_field}.")
            plt.close(fig)
            return None

        df_plot = _clean_xy(df, x_field, y_field)
        if df_plot.empty:
            print(f" Plot {idx}: cleaned bar data is empty, skipping.")
            plt.close(fig)
            return None

        # Truncate labels to avoid overcrowding
        x_vals = [_truncate_label(v) for v in df_plot[x_field]]
        y_vals = df_plot[y_field].values

        n = len(df_plot)
        cmap = plt.cm.get_cmap("tab20", max(n, 3))
        colors = [cmap(i) for i in range(n)]

        bars = ax.bar(x_vals, y_vals, color=colors)
        plt.xticks(rotation=45, ha="right")
        _style_axes(ax, reason, x_field, y_field)

        from matplotlib.patches import Patch
        handles = [Patch(color=colors[i], label=x_vals[i]) for i in range(n)]
        ax.legend(
            handles=handles,
            title="Game",
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            borderaxespad=0.0,
        )

        _annotate_bar_values(ax, bars)

    # LINE
    elif chart_type == "line":
        if x_field not in df.columns or y_field not in df.columns:
            print(f" Plot {idx}: line needs x_field={x_field}, y_field={y_field}.")
            plt.close(fig)
            return None

        df_plot = _clean_xy(df, x_field, y_field)
        if df_plot.empty:
            print(f" Plot {idx}: cleaned line data is empty, skipping.")
            plt.close(fig)
            return None

        x_vals = df_plot[x_field]
        if x_vals.dtype == "object":
            x_labels = [_truncate_label(v) for v in x_vals]
        else:
            x_labels = x_vals

        ax.plot(
            x_labels,
            df_plot[y_field],
            color=COLOR_PRIMARY,
            marker="o",
            label=y_field,
        )
        plt.xticks(rotation=45, ha="right")
        _style_axes(ax, reason, x_field, y_field)
        ax.legend()

    # SCATTER
    elif chart_type == "scatter":
        if x_field not in df.columns or y_field not in df.columns:
            print(f" Plot {idx}: scatter needs x_field={x_field}, y_field={y_field}.")
            plt.close(fig)
            return None

        df_plot = _clean_xy(df, x_field, y_field)
        if df_plot.empty:
            print(f" Plot {idx}: cleaned scatter data is empty, skipping.")
            plt.close(fig)
            return None

        ax.scatter(
            df_plot[x_field],
            df_plot[y_field],
            color=COLOR_PRIMARY,
            alpha=0.7,
            edgecolor="black",
            label=f"{y_field} vs {x_field}",
        )
        _style_axes(ax, reason, x_field, y_field)
        ax.legend()

    # HISTOGRAM
    elif chart_type == "histogram":
        col = x_field or y_field
        if col not in df.columns:
            print(f" Plot {idx}: histogram needs a numeric column '{col}'.")
            plt.close(fig)
            return None

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

    # BOX
    elif chart_type == "box":
        col = y_field or x_field
        if col not in df.columns:
            print(f" Plot {idx}: boxplot needs a numeric column '{col}'.")
            plt.close(fig)
            return None

        ax.boxplot(df[col].dropna(), patch_artist=True)
        for patch in ax.artists:
            patch.set_facecolor(COLOR_PRIMARY)
        _style_axes(ax, reason, "", col)

    # PIE
    elif chart_type == "pie":
        if x_field not in df.columns or y_field not in df.columns:
            print(f" Plot {idx}: pie chart needs x_field={x_field} and y_field={y_field}.")
            plt.close(fig)
            return None

        labels = [_truncate_label(v) for v in df[x_field]]
        values = df[y_field].values

        ax.pie(
            values,
            labels=labels,
            autopct="%1.1f%%",
            startangle=140,
        )
        ax.set_title(reason)
        ax.axis("equal")

    else:
        print(f" Plot {idx}: unknown chart_type '{chart_type}', skipping.")
        plt.close(fig)
        return None

    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode("utf-8")

    return {
        "chart_type": chart_type,
        "reason": reason,
        "image_base64": image_base64,
    }


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    # run agent
    resp = AGENT.invoke({"input": req.message})
    full_text = resp["output"]

    # split answer vs VIS_PLAN_JSON
    main_text, plan = split_answer_and_vis_plan(full_text)

    plot_results: List[Dict[str, Any]] = []
    if plan is not None:
        plots_cfg = plan.get("plots") or []
        for idx, p in enumerate(plots_cfg, start=1):
            rendered = render_plot_to_base64(p, idx)
            if rendered is not None:
                plot_results.append(rendered)

    return ChatResponse(
        answer=main_text,
        raw_output=full_text,
        plots=plot_results,
    )