import json
from typing import Optional, Tuple, Dict, Any, List

import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from sqlalchemy import create_engine

from agent import build_agent

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


def split_answer_and_vis_plan(text):
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


def _style_axes(
    ax: plt.Axes, title: str, xlabel: Optional[str], ylabel: Optional[str]
) -> None:
    ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)

    ax.grid(True, linestyle="--", alpha=0.1)


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


def draw_single_plot(plot_cfg: Dict[str, Any], idx: int):
    chart_type = plot_cfg.get("chart_type")
    sql = plot_cfg.get("sql_query")
    x_field = plot_cfg.get("x_field")
    y_field = plot_cfg.get("y_field")
    reason = plot_cfg.get("reason", f"Plot {idx}")

    print(f"\n[Plot {idx}] {reason}")
    if not sql:
        print(f"[WARN] Plot {idx}: sql_query is empty, skipping.")
        return

    try:
        df = pd.read_sql_query(sql, ENGINE)
    except Exception as e:
        print(f"[ERROR] Plot {idx}: failed to run SQL.")
        print("Error:", e)
        print("SQL:", sql)
        return

    if df.empty:
        print(f"[INFO] Plot {idx}: SQL returned no rows, skipping.")
        print("SQL:", sql)
        return

    print(f"[INFO] Plot {idx}: Data columns:", df.columns.tolist())

    fig, ax = plt.subplots()

    if chart_type == "bar":
        if x_field not in df.columns or y_field not in df.columns:
            print(
                f"[WARN] Plot {idx}: bar chart needs x_field={x_field} and "
                f"y_field={y_field} in columns."
            )
            plt.close(fig)
            return

        n = len(df)
        cmap = plt.cm.get_cmap("tab20", max(n, 3))
        colors = [cmap(i) for i in range(n)]

        bars = ax.bar(
            df[x_field],
            df[y_field],
            color=colors,
        )
        plt.xticks(rotation=15, ha="right")

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
                f"[WARN] Plot {idx}: line chart needs x_field={x_field} and "
                f"y_field={y_field} in columns."
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
                f"[WARN] Plot {idx}: scatter chart needs x_field={x_field} and "
                f"y_field={y_field} in columns."
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
            print(
                f"[WARN] Plot {idx}: histogram needs a numeric column '{col}' "
                "in the result."
            )
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
            print(
                f"[WARN] Plot {idx}: boxplot needs a numeric column '{col}' "
                "in the result."
            )
            plt.close(fig)
            return

        ax.boxplot(df[col].dropna(), patch_artist=True)
        for patch in ax.artists:
            patch.set_facecolor(COLOR_PRIMARY)
        _style_axes(ax, reason, "", col)

    elif chart_type == "pie":
        if x_field not in df.columns or y_field not in df.columns:
            print(
                f"[WARN] Plot {idx}: pie chart needs x_field={x_field} and "
                f"y_field={y_field} in columns."
            )
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
        print(f"[WARN] Plot {idx}: unknown chart_type '{chart_type}', skipping.")
        plt.close(fig)
        return

    plt.tight_layout()
    plt.show()


def visualize_from_plan(plan: Dict[str, Any]):
    plots: List[Dict[str, Any]] = plan.get("plots") or []
    if not plots:
        print("[INFO] No plots proposed in VIS_PLAN_JSON.")
        return

    for idx, p in enumerate(plots, start=1):
        draw_single_plot(p, idx)


def main():
    agent = build_agent()

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

        print("\n=== Agent Answer ===\n")
        print(main_text)
        print("\n=====================\n")

        if plan is not None:
            visualize_from_plan(plan)
        else:
            print("[INFO] No VIS_PLAN_JSON found; skipping visualization.")

        print("-*" * 50)


if __name__ == "__main__":
    main()
