# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Reconstruct and inspect the exact model input for the SWE-bench eval pipeline.

Shows what the model actually sees after vLLM applies the chat template with tools=.
Compares the eval pipeline (harbor/mini-swe-agent) vs the training pipeline (phitrain).
Generates a PDF report.
"""

import argparse
import platform
import re
import sys
import textwrap
from datetime import datetime
from pathlib import Path

from datasets import load_dataset
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from phitrain.recipes.swe_agent.swe_agent.message_formatter import create_miniswe_message_formatter
from phitrain.recipes.swe_agent.swe_agent.tools import SWE_TOOL_SCHEMAS

# --- mini-swe-agent's BASH_TOOL (copied from mini-swe-agent source) ---
MSWEA_BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Execute a bash command",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to execute",
                }
            },
            "required": ["command"],
        },
    },
}

# --- mini-swe-agent's default system & instance templates (from default.yaml) ---
MSWEA_SYSTEM_TEMPLATE = """\
You are a helpful assistant that can interact with a computer.

Your response must contain exactly ONE bash code block with ONE command (or commands connected with && or ||).
Include a THOUGHT section before your command where you explain your reasoning process.
Format your response as shown in <format_example>.

<format_example>
Your reasoning and analysis here. Explain why you want to perform the action.

```mswea_bash_command
your_command_here
```
</format_example>

Failure to follow these rules will cause your response to be rejected."""

MSWEA_INSTANCE_TEMPLATE = """\
Please solve this issue: {task}

You can execute bash commands and edit files to implement the necessary changes.

## Recommended Workflow

This workflow should be done step-by-step so that you can iterate on your changes and any possible problems.

1. Analyze the codebase by finding and reading relevant files
2. Create a script to reproduce the issue
3. Edit the source code to resolve the issue
4. Verify your fix works by running your script again
5. Test edge cases to ensure your fix is robust
6. Submit your changes and finish your work by issuing the following command: `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`.
   Do not combine it with any other command. <important>After this command, you cannot continue working on this task.</important>

## Important Rules

1. Every response must contain exactly one action
2. The action must be enclosed in triple backticks
3. Directory or environment variable changes are not persistent. Every action is executed in a new subshell.
   However, you can prefix any action with `MY_ENV_VAR=MY_VALUE cd /path/to/working/dir && ...` or write/load environment variables from files

<system_information>
{system} {release} {version} {machine}
</system_information>"""

MODEL_PATH = str(PROJECT_ROOT / "_ckpts" / "Qwen3-14B-modified")
CONFIG_PATH = str(PROJECT_ROOT / "phitrain" / "recipes" / "swe_agent" / "configs" / "swebench.yaml")


def render_mswea_system(uname) -> str:
    return MSWEA_SYSTEM_TEMPLATE


def render_mswea_instance(task: str, uname) -> str:
    return MSWEA_INSTANCE_TEMPLATE.format(
        task=task,
        system=uname.system,
        release=uname.release,
        version=uname.version,
        machine=uname.machine,
    )


def build_eval_prompt(tokenizer, task_text: str) -> dict:
    """Build the exact prompt as the eval pipeline (harbor/mini-swe-agent) would."""
    uname = platform.uname()
    system_content = render_mswea_system(uname)
    user_content = render_mswea_instance(task_text, uname)

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]
    tools = [MSWEA_BASH_TOOL]

    # Render via chat template WITH tools= (what vLLM does internally)
    rendered = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=False,
        add_generation_prompt=True,
    )

    return {
        "system_content": system_content,
        "user_content": user_content,
        "tools": tools,
        "rendered_prompt": rendered,
    }


def build_train_prompt(tokenizer, task_text: str) -> dict:
    """Build the exact prompt as the training pipeline (phitrain) would."""
    formatter = create_miniswe_message_formatter(CONFIG_PATH, tokenizer=tokenizer)
    messages = formatter(task_text)

    # Render via chat template WITHOUT tools= (training bakes them into system msg)
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    return {
        "system_content": messages[0]["content"],
        "user_content": messages[1]["content"],
        "tools": None,
        "rendered_prompt": rendered,
    }


# ---------------------------------------------------------------------------
# PDF generation
# ---------------------------------------------------------------------------

def _esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _make_table(data, col_widths, header=True):
    t = Table(data, colWidths=col_widths, repeatRows=1 if header else 0)
    style_cmds = [
        ("FONTNAME", (0, 0), (-1, -1), "Courier"),
        ("FONTSIZE", (0, 0), (-1, -1), 7),
        ("LEADING", (0, 0), (-1, -1), 9),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]
    if header:
        style_cmds += [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 8),
        ]
    for i in range(1, len(data)):
        if i % 2 == 0:
            style_cmds.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#ecf0f1")))
    t.setStyle(TableStyle(style_cmds))
    return t


def _code_block(text: str, max_width: int = 115) -> Preformatted:
    """Render text as a monospace code block, wrapping long lines."""
    wrapped = []
    for line in text.splitlines():
        if len(line) > max_width:
            wrapped.extend(textwrap.wrap(line, max_width, break_long_words=True, break_on_hyphens=False))
        else:
            wrapped.append(line)
    style = ParagraphStyle("Code", fontName="Courier", fontSize=5.5, leading=7,
                            textColor=colors.HexColor("#2c3e50"),
                            backColor=colors.HexColor("#f8f9fa"),
                            borderPadding=4)
    return Preformatted("\n".join(wrapped), style)


def _diff_table(left_lines, right_lines, left_label, right_label, col_w):
    max_lines = max(len(left_lines), len(right_lines))
    data = [["", left_label, right_label]]
    row_colors = []
    for i in range(max_lines):
        l_text = left_lines[i] if i < len(left_lines) else ""
        r_text = right_lines[i] if i < len(right_lines) else ""
        match = l_text.rstrip() == r_text.rstrip()
        data.append(["" if match else "DIFF", l_text, r_text])
        row_colors.append(None if match else colors.HexColor("#fadbd8"))

    t = Table(data, colWidths=[0.35 * inch, col_w, col_w], repeatRows=1)
    style_cmds = [
        ("FONTNAME", (0, 0), (-1, -1), "Courier"),
        ("FONTSIZE", (0, 0), (-1, -1), 5),
        ("LEADING", (0, 0), (-1, -1), 6.5),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.lightgrey),
        ("TOPPADDING", (0, 0), (-1, -1), 1),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
        ("LEFTPADDING", (0, 0), (-1, -1), 2),
        ("RIGHTPADDING", (0, 0), (-1, -1), 2),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 6),
    ]
    for i, c in enumerate(row_colors):
        if c:
            style_cmds.append(("BACKGROUND", (0, i + 1), (-1, i + 1), c))
            style_cmds.append(("TEXTCOLOR", (0, i + 1), (0, i + 1), colors.red))
    t.setStyle(TableStyle(style_cmds))
    return t


def generate_pdf(pdf_path: str, task, eval_prompt, train_prompt, tokenizer) -> None:
    doc = SimpleDocTemplate(
        pdf_path, pagesize=letter,
        leftMargin=0.4 * inch, rightMargin=0.4 * inch,
        topMargin=0.4 * inch, bottomMargin=0.4 * inch,
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("Title2", parent=styles["Title"], fontSize=16, spaceAfter=6)
    h1 = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=13, spaceBefore=14, spaceAfter=6,
                         textColor=colors.HexColor("#2c3e50"))
    h2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=10, spaceBefore=8, spaceAfter=4,
                         textColor=colors.HexColor("#34495e"))
    note = ParagraphStyle("Note", parent=styles["Normal"], fontSize=8, leading=10,
                           fontName="Helvetica", textColor=colors.HexColor("#7f8c8d"))
    body_text = ParagraphStyle("BodyText2", parent=styles["Normal"], fontSize=8, leading=10)

    page_w = letter[0] - 0.8 * inch
    col_w = (page_w - 0.35 * inch) / 2

    elements = []

    # -- Title --
    elements.append(Paragraph("SWE-bench Eval: Model Input Inspection Report", title_style))
    elements.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", note))
    elements.append(Spacer(1, 12))

    # -- Section 1: Task info --
    elements.append(Paragraph("1. Task", h1))
    info_data = [
        ["Field", "Value"],
        ["Instance ID", task["instance_id"]],
        ["Repo", task["repo"]],
        ["Model", MODEL_PATH.split("/")[-1]],
    ]
    elements.append(_make_table(info_data, [1.5 * inch, page_w - 1.5 * inch]))
    elements.append(Spacer(1, 8))

    # -- Section 2: Pipeline comparison --
    elements.append(Paragraph("2. Pipeline Comparison", h1))
    cmp_data = [
        ["", "Eval (harbor / mini-swe-agent)", "Training (phitrain)"],
        ["System message", "Plain text template\n(no tool schemas)", "Tool schemas baked in\nvia apply_chat_template(tools=)"],
        ["tools= in API call", "YES — tools=[BASH_TOOL]", "NO"],
        ["Who injects tool schemas\ninto the prompt?", "vLLM (via chat template\nwhen it sees tools=)", "Formatter at data prep time\n(before vLLM)"],
        ["Who parses tool calls\nfrom the response?", "vLLM's qwen3_xml parser\n→ tool_calls field", "Qwen35XMLParser\non raw content"],
    ]
    elements.append(_make_table(cmp_data, [2.0 * inch, (page_w - 2.0 * inch) / 2, (page_w - 2.0 * inch) / 2]))
    elements.append(Spacer(1, 12))

    # -- Section 3: Eval pipeline — messages BEFORE chat template --
    elements.append(Paragraph("3. Eval Pipeline: Messages Before Chat Template", h1))

    elements.append(Paragraph("3a. System Message (plain text, no tool schemas)", h2))
    elements.append(_code_block(eval_prompt["system_content"]))
    elements.append(Spacer(1, 6))

    elements.append(Paragraph("3b. User Message (rendered instance template)", h2))
    elements.append(_code_block(eval_prompt["user_content"][:3000]))
    elements.append(Spacer(1, 6))

    elements.append(Paragraph("3c. tools= Parameter (passed alongside messages)", h2))
    import json
    elements.append(_code_block(json.dumps(eval_prompt["tools"], indent=2)))
    elements.append(Spacer(1, 12))

    # -- Section 4: Eval pipeline — final rendered prompt --
    elements.append(PageBreak())
    elements.append(Paragraph("4. Eval Pipeline: Final Rendered Prompt (what the model sees)", h1))
    elements.append(Paragraph(
        "This is the output of tokenizer.apply_chat_template(messages, tools=[BASH_TOOL]). "
        "This is exactly what vLLM feeds to the model. Note how vLLM's chat template "
        "renders the tool schema into the system block.",
        note))
    elements.append(Spacer(1, 6))
    elements.append(_code_block(eval_prompt["rendered_prompt"]))
    elements.append(Spacer(1, 12))

    # -- Section 5: Training pipeline — final rendered prompt --
    elements.append(PageBreak())
    elements.append(Paragraph("5. Training Pipeline: Final Rendered Prompt (what the model sees)", h1))
    elements.append(Paragraph(
        "This is the output of tokenizer.apply_chat_template(messages) — no tools= parameter. "
        "The tool schemas were already baked into the system message content by the formatter.",
        note))
    elements.append(Spacer(1, 6))
    elements.append(_code_block(train_prompt["rendered_prompt"]))
    elements.append(Spacer(1, 12))

    # -- Section 6: Side-by-side diff of final prompts --
    elements.append(PageBreak())
    elements.append(Paragraph("6. Side-by-Side: Final Rendered Prompts", h1))
    elements.append(Paragraph(
        "Red rows show where the two prompts diverge. This reveals the actual difference "
        "in what the model sees between eval and training.",
        note))
    elements.append(Spacer(1, 6))

    wrap_w = 80
    def wrap_lines(text):
        result = []
        for line in text.splitlines():
            if len(line) > wrap_w:
                result.extend(textwrap.wrap(line, wrap_w, break_long_words=True, break_on_hyphens=False))
            else:
                result.append(line)
        return result

    elements.append(_diff_table(
        wrap_lines(eval_prompt["rendered_prompt"]),
        wrap_lines(train_prompt["rendered_prompt"]),
        "Eval (harbor/mini-swe-agent)",
        "Training (phitrain)",
        col_w,
    ))
    elements.append(Spacer(1, 12))

    # -- Section 7: Token count comparison --
    elements.append(Paragraph("7. Token Counts", h1))
    eval_tokens = tokenizer.encode(eval_prompt["rendered_prompt"])
    train_tokens = tokenizer.encode(train_prompt["rendered_prompt"])
    tok_data = [
        ["", "Eval", "Training", "Diff"],
        ["Chars", str(len(eval_prompt["rendered_prompt"])),
         str(len(train_prompt["rendered_prompt"])),
         str(len(eval_prompt["rendered_prompt"]) - len(train_prompt["rendered_prompt"]))],
        ["Tokens", str(len(eval_tokens)), str(len(train_tokens)),
         str(len(eval_tokens) - len(train_tokens))],
    ]
    elements.append(_make_table(tok_data, [1.5 * inch, 1.5 * inch, 1.5 * inch, 1.5 * inch]))
    elements.append(Spacer(1, 12))

    # -- Section 8: Key finding --
    elements.append(Paragraph("8. Key Finding", h1))

    # Check if the tool schema sections are identical
    # Extract the tool section from both rendered prompts
    tool_section_re = re.compile(r"# Tools.*?(?=\n\n[^#]|\nYou are)", re.DOTALL)
    eval_tool_match = tool_section_re.search(eval_prompt["rendered_prompt"])
    train_tool_match = tool_section_re.search(train_prompt["rendered_prompt"])

    if eval_tool_match and train_tool_match:
        tools_identical = eval_tool_match.group(0).strip() == train_tool_match.group(0).strip()
        elements.append(Paragraph(
            f"Tool schema sections are: <b>{'IDENTICAL' if tools_identical else 'DIFFERENT'}</b>",
            body_text))
    else:
        elements.append(Paragraph("Could not extract tool schema sections for comparison.", body_text))

    elements.append(Spacer(1, 6))
    elements.append(Paragraph(
        "Both pipelines use the same chat template to render the prompt. The tool schemas end up "
        "in the same location (inside the system block) regardless of whether they were baked in by "
        "the formatter (training) or injected by vLLM via tools= (eval). The difference is in the "
        "system/user message content — different templates produce different instructions.",
        note))

    doc.build(elements)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect exact model input for SWE-bench eval")
    parser.add_argument("--instance-id", type=str, default=None)
    parser.add_argument("--model-path", type=str, default=MODEL_PATH)
    parser.add_argument("--pdf", type=str, default="eval/swe_bench/eval_prompt_report.pdf")
    args = parser.parse_args()

    print("Loading dataset...")
    dataset = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")

    if args.instance_id:
        matches = [row for row in dataset if row["instance_id"] == args.instance_id]
        if not matches:
            print(f"ERROR: instance_id '{args.instance_id}' not found")
            sys.exit(1)
        task = matches[0]
    else:
        task = dataset[0]

    print(f"Task: {task['instance_id']} ({task['repo']})")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    print("Building eval pipeline prompt...")
    eval_prompt = build_eval_prompt(tokenizer, task["problem_statement"])

    print("Building training pipeline prompt...")
    train_prompt = build_train_prompt(tokenizer, task["problem_statement"])

    # Print key info to stdout
    print(f"\n{'=' * 80}")
    print("  EVAL PIPELINE — Final rendered prompt ({} chars, {} tokens)".format(
        len(eval_prompt["rendered_prompt"]),
        len(tokenizer.encode(eval_prompt["rendered_prompt"])),
    ))
    print(f"{'=' * 80}")
    print(eval_prompt["rendered_prompt"])

    print(f"\n{'=' * 80}")
    print("  TRAINING PIPELINE — Final rendered prompt ({} chars, {} tokens)".format(
        len(train_prompt["rendered_prompt"]),
        len(tokenizer.encode(train_prompt["rendered_prompt"])),
    ))
    print(f"{'=' * 80}")
    print(train_prompt["rendered_prompt"])

    print(f"\nGenerating PDF report -> {args.pdf}")
    generate_pdf(args.pdf, task, eval_prompt, train_prompt, tokenizer)
    print(f"Done: {args.pdf}")


if __name__ == "__main__":
    main()
