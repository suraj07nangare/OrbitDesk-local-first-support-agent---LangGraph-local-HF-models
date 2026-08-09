from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUTPUT_PATH = Path(__file__).resolve().parents[1] / "diagrams" / "graph.png"

NODES = {
    "start": (1.0, 11.4, "START", "#455a64"),
    "triage": (1.0, 9.9, "triage\n(rule-based signals)", "#1e88e5"),
    "retrieve": (3.8, 8.4, "retrieve\n(embedding search)", "#1e88e5"),
    "safe_response": (-2.4, 6.8, "safe_response\n(out of scope /\npolicy violation)", "#e53935"),
    "clarify": (1.4, 6.4, "clarify\n(ask for detail)", "#fb8c00"),
    "generate": (6.4, 6.4, "generate\n(local LLM)", "#1e88e5"),
    "verify": (6.4, 4.0, "verify\n(groundedness +\nrule checks)", "#1e88e5"),
    "safe_failure": (4.2, 1.8, "safe_failure\n(max retries used)", "#e53935"),
    "finalize": (0.8, 0.2, "finalize\n(schema-valid output)", "#43a047"),
    "end": (0.8, -1.4, "END", "#455a64"),
}

EDGES = [
    ("start", "triage", "", 0.08, None),
    ("triage", "retrieve", "no policy\nviolation", 0.08, None),
    ("triage", "safe_response", "policy\nviolation", 0.08, None),
    ("retrieve", "safe_response", "no relevant\nevidence", -0.15, (0.6, 7.9)),
    ("retrieve", "clarify", "vague /\nlow confidence", 0.1, None),
    ("retrieve", "generate", "escalation signal\nor answerable", 0.08, None),
    ("generate", "verify", "", 0.08, None),
    ("verify", "generate", "verification failed\n(attempts < 2) [retry]", 0.4, (8.4, 5.2)),
    ("verify", "safe_failure", "verification failed\n(attempts = 2)", 0.1, None),
    ("verify", "finalize", "verification passed", -0.55, (6.7, 1.6)),
    ("clarify", "finalize", "", 0.08, None),
    ("safe_response", "finalize", "", -0.08, None),
    ("safe_failure", "finalize", "", 0.08, None),
    ("finalize", "end", "", 0.08, None),
]


def draw() -> None:
    fig, ax = plt.subplots(figsize=(14, 14))
    ax.set_xlim(-4.0, 8.6)
    ax.set_ylim(-2.4, 12.2)
    ax.axis("off")
    ax.set_title(
        "OrbitDesk Support Agent - LangGraph Workflow",
        fontsize=18,
        fontweight="bold",
        pad=20,
    )

    boxes = {}
    for key, (x, y, label, color) in NODES.items():
        width, height = 2.35, 0.95
        box = FancyBboxPatch(
            (x - width / 2, y - height / 2),
            width,
            height,
            boxstyle="round,pad=0.08,rounding_size=0.12",
            linewidth=1.4,
            edgecolor="#212121",
            facecolor=color,
            alpha=0.92,
        )
        ax.add_patch(box)
        ax.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            fontsize=10.5,
            color="white",
            fontweight="bold",
        )
        boxes[key] = (x, y, width, height)

    for src, dst, label, rad, label_pos in EDGES:
        x1, y1, w1, h1 = boxes[src]
        x2, y2, w2, h2 = boxes[dst]
        is_retry = src == "verify" and dst == "generate"

        arrow = FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            connectionstyle=f"arc3,rad={rad}",
            arrowstyle="-|>",
            mutation_scale=16,
            linewidth=1.6,
            color="#c62828" if is_retry else "#37474f",
            shrinkA=34,
            shrinkB=34,
            zorder=1,
        )
        ax.add_patch(arrow)

        if label:
            if label_pos is not None:
                mid_x, mid_y = label_pos
            else:
                mid_x = (x1 + x2) / 2
                mid_y = (y1 + y2) / 2
            ax.text(
                mid_x,
                mid_y,
                label,
                ha="center",
                va="center",
                fontsize=8.2,
                color="#c62828" if is_retry else "#263238",
                bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="#cfd8dc", alpha=0.92),
                zorder=2,
            )

    legend_items = [
        ("#1e88e5", "Processing node (deterministic or model call)"),
        ("#fb8c00", "Clarification path"),
        ("#e53935", "Safe / refusal path"),
        ("#43a047", "Final schema-valid output"),
    ]
    for i, (color, text) in enumerate(legend_items):
        ly = -2.1
        lx = -3.8 + i * 3.2
        ax.add_patch(plt.Rectangle((lx, ly - 0.1), 0.22, 0.22, facecolor=color, edgecolor="#212121"))
        ax.text(lx + 0.32, ly, text, fontsize=8, va="center")

    fig.tight_layout()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=200)
    print(f"Saved diagram to {OUTPUT_PATH}")


if __name__ == "__main__":
    draw()
