"""Build the README chart and animated summary from saved result files. No number is typed by hand."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = ROOT / "docs" / "media"
INK, MUTED, ACCENT, GOOD, BAD, LINE = "#1c2430", "#5d6877", "#24477f", "#17693f", "#a42a22", "#d8dee6"


def load(path: str) -> dict:
    return json.loads((RESULTS / path).read_text())


full = load("full/metrics.json")["conditions"]
weak = load("weak/metrics.json")["conditions"]
training = load("full/train_log.json")
rollout = load("rolling_update.json")
kafka = load("kafka_restart.json")
data = load("data_text2sql.json")
base, tuned, short = (
    full["base"]["execution_accuracy"],
    full["qlora"]["execution_accuracy"],
    weak["qlora"]["execution_accuracy"],
)

fig, ax = plt.subplots(figsize=(7.2, 3.1), dpi=150)
labels = [f"Short run\n({training['train_examples'] * 2 // 100:,} rows, blocked)", "Base model", "Full QLoRA run\n(promoted)"]
bars = ax.barh(labels, [short, base, tuned], color=[BAD, "#9aa5b1", GOOD], height=0.58)
for bar, value in zip(bars, [short, base, tuned], strict=True):
    ax.text(value + 1, bar.get_y() + bar.get_height() / 2, f"{value}%", va="center", color=INK, fontweight="bold")
ax.set_xlim(0, 85)
ax.set_xlabel(f"Execution accuracy on {full['base']['n']} held-out questions", color=MUTED)
ax.spines[["top", "right", "left"]].set_visible(False)
ax.tick_params(axis="y", length=0)
fig.tight_layout()
fig.savefig(OUT / "accuracy.png")

STAGES = [
    ("Data", "Spark", f"{sum(data['counts'].values()):,} rows versioned"),
    ("Train", "QLoRA", f"{training['train_minutes']} min on a T4"),
    ("Evaluate", "SQL runs", f"{base}% to {tuned}%"),
    ("Gate", "Airflow", f"{short}% blocked, {tuned}% promoted"),
    ("Track", "MLflow", "tied to the data hash"),
    ("Serve", "Kubernetes", f"{rollout['failed']} of {rollout['requests']} failed in an update"),
    ("Monitor", "Kafka", f"{kafka['stored_at_end']} of {kafka['sent']} stored after a kill"),
]
W, H, BOX_H = 860, 470, 44


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "Helvetica.ttc"
    return ImageFont.truetype(f"/System/Library/Fonts/{name}", size, index=1 if bold else 0)


def frame(active: int) -> Image.Image:
    image = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(image)
    draw.text((28, 20), "LLM pipeline: measured results", fill=INK, font=font(24, True))
    draw.text((28, 54), "Animated summary built from this repository's result files", fill=MUTED, font=font(15))
    for i, (stage, tool, result) in enumerate(STAGES):
        y = 92 + i * 51
        on = i <= active
        draw.rounded_rectangle((28, y, 832, y + BOX_H), 7, fill="#eef2f8" if on else "#f7f8fa", outline=ACCENT if i == active else LINE, width=2)
        draw.text((46, y + 12), stage, fill=INK if on else "#aab2bc", font=font(18, True))
        draw.text((190, y + 13), tool, fill=MUTED if on else "#c3c9d0", font=font(16))
        if on:
            draw.text((350, y + 12), result, fill=GOOD, font=font(18, True))
    return image


frames = [frame(i) for i in range(len(STAGES))]
frames[0].save(OUT / "pipeline.gif", save_all=True, append_images=frames[1:], duration=[1100] * 6 + [4000], loop=0, optimize=True)
