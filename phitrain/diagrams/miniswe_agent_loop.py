"""Generate a PNG diagram of the MiniSWE Agent Loop."""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

fig, ax = plt.subplots(figsize=(14, 20))
ax.set_xlim(0, 10)
ax.set_ylim(0, 22)
ax.axis("off")
fig.patch.set_facecolor("white")

# Colors
C_HEADER = "#1a1a2e"
C_SETUP = "#e8f4f8"
C_STEP = "#fff8e7"
C_LLM = "#4a90d9"
C_PARSE = "#f0f0f0"
C_EXEC = "#5cb85c"
C_ERROR = "#e74c3c"
C_STOP = "#d9534f"
C_OBS = "#f5a623"
C_REWARD = "#8e44ad"
C_NUDGE = "#f39c12"
C_TEXT = "#1a1a1a"

def box(x, y, w, h, text, fc, ec="#333", fontsize=10, fontweight="normal", fontcolor=C_TEXT, style="round", alpha=1.0):
    fancy = mpatches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"{style},pad=0.15",
        facecolor=fc, edgecolor=ec, linewidth=1.5, alpha=alpha,
    )
    ax.add_patch(fancy)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, fontweight=fontweight, color=fontcolor, wrap=True)

def arrow(x1, y1, x2, y2, color="#333", style="-|>", lw=1.5):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw))

def label(x, y, text, fontsize=8, color="#666", ha="center"):
    ax.text(x, y, text, ha=ha, va="center", fontsize=fontsize, color=color, style="italic")

# === Title ===
ax.text(5, 21.3, "MiniSWE Agent Loop", ha="center", va="center",
        fontsize=18, fontweight="bold", color=C_HEADER)
ax.text(5, 20.9, "Multi-turn RL rollout with real code execution", ha="center", va="center",
        fontsize=10, color="#666")

# === Setup ===
box(2.5, 19.7, 5, 0.7, "Create Docker/Bubblewrap Environment\nfor SWE-bench instance", C_SETUP, fontsize=9)
arrow(5, 19.7, 5, 19.3, color="#555")

# === Loop boundary ===
loop_rect = mpatches.FancyBboxPatch(
    (0.8, 5.8), 8.4, 13.3,
    boxstyle="round,pad=0.2", facecolor="#fafafa", edgecolor="#999",
    linewidth=2, linestyle="--", alpha=0.5,
)
ax.add_patch(loop_rect)
ax.text(1.2, 18.85, "for step in range(max_steps)", fontsize=9,
        fontweight="bold", color="#555", style="italic")

# === Context check ===
box(2.5, 17.8, 5, 0.65, "Context length ≥ max?", C_PARSE, fontsize=9)
arrow(5, 17.8, 5, 17.35, color="#555")
label(5.3, 17.55, "no", fontsize=8)

# Arrow to STOP (right side)
ax.annotate("", xy=(8.8, 18.1), xytext=(7.5, 18.1),
            arrowprops=dict(arrowstyle="-|>", color=C_STOP, lw=1.5))
label(8.1, 18.35, "yes", color=C_STOP)
box(8.2, 17.8, 1.3, 0.55, "STOP", C_STOP, fontsize=9, fontweight="bold", fontcolor="white", ec=C_STOP)

# === Context nudge ===
box(2.5, 16.6, 5, 0.65, "Context ≥ 80%? → Inject nudge\n\"finalize & submit now\"", "#fef3cd", fontsize=8, ec=C_NUDGE)
arrow(5, 16.6, 5, 16.15, color="#555")

# === LLM Generate ===
box(2.5, 15.1, 5, 0.9, "LLM Generate\n(rollout_client.generate)\ntokens + logprobs", C_LLM, fontsize=9, fontweight="bold", fontcolor="white", ec="#3a7bc8")
arrow(5, 15.1, 5, 14.65, color="#555")

# === Add to Interaction ===
box(2.5, 13.9, 5, 0.65, "Append assistant response\nto Interaction (tokens, masks=1, logprobs)", C_PARSE, fontsize=8)
arrow(5, 13.9, 5, 13.45, color="#555")

# === Parse Action ===
box(2.5, 12.7, 5, 0.65, "Parse action from response\nregex: ```mswea_bash_command```", C_PARSE, fontsize=8)

# Parse error branch (right)
ax.annotate("", xy=(8.8, 13.0), xytext=(7.5, 13.0),
            arrowprops=dict(arrowstyle="-|>", color=C_ERROR, lw=1.5))
label(8.1, 13.25, "0 or 2+ actions", color=C_ERROR, fontsize=7)

box(8.0, 11.8, 1.8, 1.0, "FORMAT\nERROR\nzero masks\nsend error", C_ERROR, fontsize=7, fontweight="bold", fontcolor="white", ec=C_ERROR)

# Error loops back up
ax.annotate("", xy=(9.2, 18.45), xytext=(9.2, 12.8),
            arrowprops=dict(arrowstyle="-|>", color=C_ERROR, lw=1.2,
                          connectionstyle="arc3,rad=0.15"))
label(9.55, 15.5, "retry", color=C_ERROR, fontsize=7)

# Parse success (down)
arrow(5, 12.7, 5, 12.25, color="#555")
label(4.3, 12.45, "1 action", fontsize=8, color=C_EXEC)

# === Execute ===
box(2.5, 11.2, 5, 0.9, "Execute command\nin Container", C_EXEC, fontsize=10, fontweight="bold", fontcolor="white", ec="#4cae4c")

# === Three outcomes ===

# Submitted (left)
ax.annotate("", xy=(1.0, 11.65), xytext=(2.5, 11.65),
            arrowprops=dict(arrowstyle="-|>", color=C_STOP, lw=1.5))
label(1.8, 11.9, "submit", color=C_STOP, fontsize=8)
box(0.1, 11.15, 1.0, 0.65, "STOP\n✓", C_STOP, fontsize=9, fontweight="bold", fontcolor="white", ec=C_STOP)

# Timeout/error (right)
ax.annotate("", xy=(8.2, 11.65), xytext=(7.5, 11.65),
            arrowprops=dict(arrowstyle="-|>", color=C_ERROR, lw=1.2))
label(7.85, 11.9, "timeout", color="#888", fontsize=7)

# Normal output (down)
arrow(5, 11.2, 5, 10.75, color="#555")
label(5.5, 10.95, "output", fontsize=8, color="#555")

# Timeout merges into observation
ax.annotate("", xy=(7.0, 10.3), xytext=(8.2, 11.15),
            arrowprops=dict(arrowstyle="-|>", color="#888", lw=1.0))

# === Observation ===
box(2.5, 9.7, 5, 0.7, "Format observation via template\nAppend as \"user\" turn", C_OBS, fontsize=9, fontcolor="white", fontweight="bold", ec="#e09500")

# Loop back arrow
ax.annotate("", xy=(1.5, 18.45), xytext=(1.5, 10.0),
            arrowprops=dict(arrowstyle="-|>", color="#555", lw=1.5,
                          connectionstyle="arc3,rad=-0.15"))
label(0.9, 14.0, "next\nstep", fontsize=8, color="#555")

# === Exit conditions ===
box(1.5, 8.4, 7, 0.7, "Exit: context full  |  max steps reached  |  submit command", "#f0f0f0",
    fontsize=9, ec="#999")
arrow(5, 8.4, 5, 7.95, color="#555")

# === Reward ===
box(2.5, 7.0, 5, 0.7, "RewardManager scores interaction\n(run SWE-bench tests on patch)", C_REWARD, fontsize=9, fontcolor="white", fontweight="bold", ec="#7b3fa0")
arrow(5, 7.0, 5, 6.55, color="#555")

# === Cleanup ===
box(2.5, 5.8, 5, 0.65, "Cleanup: remove container", C_SETUP, fontsize=9)

# === Legend ===
legend_y = 4.8
ax.text(1.0, legend_y, "Key:", fontsize=9, fontweight="bold", color=C_TEXT)
legend_items = [
    (C_LLM, "LLM inference"),
    (C_EXEC, "Code execution"),
    (C_OBS, "Observation"),
    (C_ERROR, "Error handling"),
    (C_REWARD, "Reward scoring"),
]
for i, (color, desc) in enumerate(legend_items):
    x = 1.0 + i * 1.8
    rect = mpatches.FancyBboxPatch((x, legend_y - 0.6), 0.3, 0.25,
                                     boxstyle="round,pad=0.05", facecolor=color, edgecolor=color)
    ax.add_patch(rect)
    ax.text(x + 0.45, legend_y - 0.47, desc, fontsize=7, va="center", color=C_TEXT)

# === Notes ===
notes = [
    "• Masks are zeroed on format errors → RL loss ignores broken turns",
    "• Full token sequence with logprobs accumulates for policy gradient training",
    "• Context nudge at 80% prevents wasted compute on truncated episodes",
]
for i, note in enumerate(notes):
    ax.text(0.5, 3.6 - i * 0.35, note, fontsize=7.5, color="#555", va="center")

plt.tight_layout()
plt.savefig("/home/abgoswam/_hackerreborn/aifsdk/diagrams/miniswe_agent_loop.png",
            dpi=200, bbox_inches="tight", facecolor="white")
print("Saved to diagrams/miniswe_agent_loop.png")
