# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Debug script: run 1 SWE-bench task for 1 turn and inspect messages/response/parsing.

Sends the same prompt to two vLLM instances for side-by-side comparison:
  - Server A (port 8000): raw mode, no tool-call parser
  - Server B (port 8001): --enable-auto-tool-choice --tool-call-parser qwen3_xml
"""

import argparse
import json
import re
import sys
import textwrap
from datetime import datetime
from pathlib import Path

from datasets import load_dataset
from openai import OpenAI
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from transformers import AutoTokenizer

# Add project root to path so phitrain imports work
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from phitrain.recipes.common.parsers import Qwen35XMLParser, check_format_warnings
from phitrain.recipes.swe_agent.swe_agent.message_formatter import create_miniswe_message_formatter
from phitrain.recipes.swe_agent.swe_agent.tools import SWE_TOOL_SCHEMAS

# Defaults
MODEL_PATH = str(PROJECT_ROOT / "_ckpts" / "Qwen3-14B-modified")
CONFIG_PATH = str(PROJECT_ROOT / "phitrain" / "recipes" / "swe_agent" / "configs" / "swebench.yaml")
SERVED_MODEL_NAME = "vllm_hosted_model"

# Two servers
SERVER_A_URL = "http://localhost:8000/v1"  # raw mode (no tool-call parser)
SERVER_B_URL = "http://localhost:8001/v1"  # --tool-call-parser qwen3_xml


def separator(title: str) -> None:
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}\n")


# ---------------------------------------------------------------------------
# PDF report generation
# ---------------------------------------------------------------------------

def _esc(text: str) -> str:
    """Escape XML special chars for reportlab Paragraphs."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _make_table(data: list[list[str]], col_widths: list[float], header: bool = True) -> Table:
    """Build a styled reportlab Table."""
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
    # Alternate row shading
    for i in range(1, len(data)):
        if i % 2 == 0:
            style_cmds.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#ecf0f1")))
    t.setStyle(TableStyle(style_cmds))
    return t


def _diff_table(left_lines: list[str], right_lines: list[str], left_label: str, right_label: str, col_w: float) -> Table:
    """Build a side-by-side diff table with row highlighting."""
    max_lines = max(len(left_lines), len(right_lines))
    data = [["", left_label, right_label]]
    row_colors = []
    for i in range(max_lines):
        l_text = left_lines[i] if i < len(left_lines) else ""
        r_text = right_lines[i] if i < len(right_lines) else ""
        match = l_text.strip() == r_text.strip()
        data.append(["" if match else "DIFF", l_text, r_text])
        row_colors.append(None if match else colors.HexColor("#fadbd8"))

    t = Table(data, colWidths=[0.4 * inch, col_w, col_w], repeatRows=1)
    style_cmds = [
        ("FONTNAME", (0, 0), (-1, -1), "Courier"),
        ("FONTSIZE", (0, 0), (-1, -1), 6),
        ("LEADING", (0, 0), (-1, -1), 7.5),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.lightgrey),
        ("TOPPADDING", (0, 0), (-1, -1), 1),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 7),
    ]
    for i, c in enumerate(row_colors):
        if c:
            style_cmds.append(("BACKGROUND", (0, i + 1), (-1, i + 1), c))
            style_cmds.append(("TEXTCOLOR", (0, i + 1), (0, i + 1), colors.red))
    t.setStyle(TableStyle(style_cmds))
    return t


def generate_pdf(pdf_path, task, args, messages, response_a, response_b,
                 raw_content_a, raw_content_b, tool_calls_b,
                 parsed_a, parsed_b, warnings_a, warnings_b,
                 agent_cmd_a, agent_cmd_b, vllm_cmd_b) -> None:
    """Generate the PDF comparison report."""

    doc = SimpleDocTemplate(
        pdf_path, pagesize=letter,
        leftMargin=0.5 * inch, rightMargin=0.5 * inch,
        topMargin=0.5 * inch, bottomMargin=0.5 * inch,
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("Title2", parent=styles["Title"], fontSize=16, spaceAfter=6)
    h1 = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=13, spaceBefore=14, spaceAfter=6,
                         textColor=colors.HexColor("#2c3e50"))
    h2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=11, spaceBefore=10, spaceAfter=4,
                         textColor=colors.HexColor("#34495e"))
    body = ParagraphStyle("Body2", parent=styles["Normal"], fontSize=8, leading=10, fontName="Courier")
    body_sm = ParagraphStyle("BodySm", parent=styles["Normal"], fontSize=7, leading=9, fontName="Courier")
    note = ParagraphStyle("Note", parent=styles["Normal"], fontSize=8, leading=10, fontName="Helvetica",
                           textColor=colors.HexColor("#7f8c8d"))

    page_w = letter[0] - 1.0 * inch  # usable width
    col_w = (page_w - 0.4 * inch) / 2  # for diff table

    elements = []

    # -- Title --
    elements.append(Paragraph("SWE-bench Debug: vLLM Parser Comparison Report", title_style))
    elements.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", note))
    elements.append(Spacer(1, 12))

    # -- Section 1: Task & Config --
    elements.append(Paragraph("1. Task &amp; Configuration", h1))
    config_data = [
        ["Parameter", "Value"],
        ["Instance ID", task["instance_id"]],
        ["Repo", task["repo"]],
        ["Model", args.model_path.split("/")[-1]],
        ["Temperature", str(args.temperature)],
        ["Seed", str(args.seed)],
        ["Max tokens", str(args.max_tokens)],
    ]
    elements.append(_make_table(config_data, [1.5 * inch, page_w - 1.5 * inch]))
    elements.append(Spacer(1, 8))

    # -- Section 2: Server Configuration --
    elements.append(Paragraph("2. Server Configuration", h1))
    server_data = [
        ["", "Server A (port 8000)", "Server B (port 8001)"],
        ["GPU", "0", "1"],
        ["Tool-call parser", "None (raw mode)", "qwen3_xml"],
        ["--enable-auto-tool-choice", "No", "Yes"],
        ["tools= in API call", "No", "Yes"],
        ["Purpose", "Training pipeline", "vLLM-native tool calling"],
    ]
    elements.append(_make_table(server_data, [1.8 * inch, (page_w - 1.8 * inch) / 2, (page_w - 1.8 * inch) / 2]))
    elements.append(Spacer(1, 8))

    # -- Section 3: Response Metadata --
    elements.append(Paragraph("3. Response Metadata", h1))
    fin_a = str(response_a.choices[0].finish_reason)
    fin_b = str(response_b.choices[0].finish_reason)
    tc_a = str(bool(response_a.choices[0].message.tool_calls))
    tc_b = str(bool(tool_calls_b))
    xml_a = str("<tool_call>" in raw_content_a)
    xml_b = str("<tool_call>" in raw_content_b)
    meta_data = [
        ["Field", "Server A", "Server B", "Match?"],
        ["finish_reason", fin_a, fin_b, "YES" if fin_a == fin_b else "DIFF"],
        ["content length", str(len(raw_content_a)), str(len(raw_content_b)),
         "YES" if len(raw_content_a) == len(raw_content_b) else "DIFF"],
        ["content has <tool_call> XML", xml_a, xml_b, "YES" if xml_a == xml_b else "DIFF"],
        ["tool_calls field populated", tc_a, tc_b, "YES" if tc_a == tc_b else "DIFF"],
    ]
    meta_table = _make_table(meta_data, [2.0 * inch, 1.5 * inch, 1.5 * inch, 0.8 * inch])
    # Highlight DIFF cells in the Match? column
    for i, row in enumerate(meta_data[1:], start=1):
        if row[3] == "DIFF":
            meta_table.setStyle(TableStyle([
                ("BACKGROUND", (3, i), (3, i), colors.HexColor("#fadbd8")),
                ("TEXTCOLOR", (3, i), (3, i), colors.red),
            ]))
    elements.append(meta_table)
    elements.append(Spacer(1, 8))

    # -- Section 4: Thinking Side-by-Side --
    elements.append(Paragraph("4. Thinking Content (Side-by-Side)", h1))
    elements.append(Paragraph(
        "Lines highlighted in red diverge between the two servers. "
        "Blank lines are shown as empty rows.",
        note))
    elements.append(Spacer(1, 4))

    # Split thinking from tool_call for Server A
    think_a = raw_content_a
    if "</think>" in think_a:
        think_a = think_a[:think_a.index("</think>") + len("</think>")]
    think_b = raw_content_b
    if "</think>" in think_b:
        think_b = think_b[:think_b.index("</think>") + len("</think>")]

    # Wrap long lines for the table
    wrap_w = 72
    def wrap_lines(text):
        result = []
        for line in text.splitlines():
            if len(line) > wrap_w:
                result.extend(textwrap.wrap(line, wrap_w, break_long_words=True, break_on_hyphens=False))
            else:
                result.append(line)
        return result

    elements.append(_diff_table(wrap_lines(think_a), wrap_lines(think_b),
                                "Server A (raw)", "Server B (qwen3_xml)", col_w))
    elements.append(Spacer(1, 8))

    # -- Section 5: Tool Call XML (Server A only) --
    elements.append(Paragraph("5. Tool Call XML in Content", h1))

    elements.append(Paragraph("Server A: &lt;tool_call&gt; XML present in content field", h2))
    tool_xml_a = ""
    if "<tool_call>" in raw_content_a:
        idx = raw_content_a.index("<tool_call>")
        tool_xml_a = raw_content_a[idx:]
    elements.append(Paragraph(_esc(tool_xml_a) if tool_xml_a else "(not found)", body_sm))
    elements.append(Spacer(1, 6))

    elements.append(Paragraph("Server B: &lt;tool_call&gt; XML stripped from content by vLLM", h2))
    elements.append(Paragraph(
        "(vLLM's qwen3_xml parser consumed the XML and moved it to the tool_calls field)",
        note))
    elements.append(Spacer(1, 8))

    # -- Section 6: Parser Results --
    elements.append(Paragraph("6. Parser Results", h1))
    parser_data = [
        ["Parser", "Applied to", "Success?", "Commands", "Error"],
        ["Qwen35XMLParser", "Server A content", "YES" if not parsed_a.error else "NO",
         str(len(parsed_a.commands)), parsed_a.error or "-"],
        ["Qwen35XMLParser", "Server B content", "YES" if not parsed_b.error else "NO",
         str(len(parsed_b.commands)), parsed_b.error or "-"],
        ["vLLM qwen3_xml", "Server B (internal)", "YES" if tool_calls_b else "NO",
         str(len(tool_calls_b)), "-"],
    ]
    pt = _make_table(parser_data, [1.3 * inch, 1.3 * inch, 0.7 * inch, 0.8 * inch, page_w - 4.1 * inch])
    # Color the success/failure cells
    for i, row in enumerate(parser_data[1:], start=1):
        bg = colors.HexColor("#d5f5e3") if row[2] == "YES" else colors.HexColor("#fadbd8")
        pt.setStyle(TableStyle([("BACKGROUND", (2, i), (2, i), bg)]))
    elements.append(pt)
    elements.append(Spacer(1, 8))

    # -- Section 7: Extracted Commands --
    elements.append(Paragraph("7. Extracted Commands", h1))

    elements.append(Paragraph("Server A / Qwen35XMLParser:", h2))
    elements.append(Paragraph(_esc(agent_cmd_a or "(none)"), body_sm))
    elements.append(Spacer(1, 6))

    elements.append(Paragraph("Server B / vLLM qwen3_xml:", h2))
    elements.append(Paragraph(_esc(vllm_cmd_b or "(none)"), body_sm))
    elements.append(Spacer(1, 6))

    elements.append(Paragraph("Server B / Qwen35XMLParser:", h2))
    elements.append(Paragraph(_esc(agent_cmd_b or "(none - parser failed, XML was stripped)"), body_sm))
    elements.append(Spacer(1, 12))

    # -- Section 8: Key Takeaway --
    elements.append(Paragraph("8. Key Takeaway", h1))
    takeaway_data = [
        ["", "content field", "tool_calls field", "Qwen35XMLParser"],
        ["Server A\n(raw mode)", "has <tool_call> XML", "(empty)", "WORKS"],
        ["Server B\n(qwen3_xml)", "XML stripped out", "populated by vLLM", "FAILS"],
    ]
    tt = _make_table(takeaway_data, [1.3 * inch, (page_w - 1.3 * inch) / 3] * 3)
    # Highlight WORKS/FAILS
    tt.setStyle(TableStyle([
        ("BACKGROUND", (3, 1), (3, 1), colors.HexColor("#d5f5e3")),
        ("BACKGROUND", (3, 2), (3, 2), colors.HexColor("#fadbd8")),
        ("TEXTCOLOR", (3, 2), (3, 2), colors.red),
    ]))
    elements.append(tt)
    elements.append(Spacer(1, 8))
    elements.append(Paragraph(
        "The two parsers are mutually exclusive: Qwen35XMLParser needs the "
        "&lt;tool_call&gt; XML in content (raw mode), while vLLM's qwen3_xml parser "
        "strips it from content and moves it to tool_calls. The training pipeline "
        "uses raw mode + Qwen35XMLParser.",
        note))

    doc.build(elements)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Debug single SWE-bench instance")
    parser.add_argument("--instance-id", type=str, default=None, help="SWE-bench instance_id (default: first task)")
    parser.add_argument("--model-path", type=str, default=MODEL_PATH, help="Path to model checkpoint")
    parser.add_argument("--config-path", type=str, default=CONFIG_PATH, help="Path to swebench.yaml config")
    parser.add_argument("--max-tokens", type=int, default=2048, help="Max tokens for generation")
    parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--pdf", type=str, default=None, help="Generate PDF report at this path")
    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # Step 1: Load 1 SWE-bench task
    # -------------------------------------------------------------------------
    separator("Step 1: Loading SWE-bench task")
    dataset = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")

    if args.instance_id:
        matches = [row for row in dataset if row["instance_id"] == args.instance_id]
        if not matches:
            print(f"ERROR: instance_id '{args.instance_id}' not found in dataset")
            sys.exit(1)
        task = matches[0]
    else:
        task = dataset[0]

    print(f"Instance ID : {task['instance_id']}")
    print(f"Repo        : {task['repo']}")
    print(f"Problem stmt: {task['problem_statement'][:200]}...")

    # -------------------------------------------------------------------------
    # Step 2: Build messages using real pipeline code
    # -------------------------------------------------------------------------
    separator("Step 2: Building messages")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    formatter = create_miniswe_message_formatter(args.config_path, tokenizer=tokenizer)
    messages = formatter(task["problem_statement"])

    print(f"System message: {len(messages[0]['content'])} chars")
    print(f"User message: {len(messages[1]['content'])} chars")

    # -------------------------------------------------------------------------
    # Step 3A: Send to Server A (raw mode, no tool-call parser)
    # -------------------------------------------------------------------------
    separator("Step 3A: Server A — raw mode (port 8000)")
    client_a = OpenAI(base_url=SERVER_A_URL, api_key="dummy")
    print("Sending request...")

    response_a = client_a.chat.completions.create(
        model=SERVED_MODEL_NAME,
        messages=messages,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        seed=args.seed,
    )
    raw_content_a = response_a.choices[0].message.content or ""
    print(f"Done. finish_reason={response_a.choices[0].finish_reason}, {len(raw_content_a)} chars")

    # -------------------------------------------------------------------------
    # Step 3B: Send to Server B (--tool-call-parser qwen3_xml)
    # -------------------------------------------------------------------------
    separator("Step 3B: Server B — qwen3_xml parser (port 8001)")
    client_b = OpenAI(base_url=SERVER_B_URL, api_key="dummy")
    print("Sending request...")

    response_b = client_b.chat.completions.create(
        model=SERVED_MODEL_NAME,
        messages=messages,
        tools=SWE_TOOL_SCHEMAS,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        seed=args.seed,
    )
    raw_content_b = response_b.choices[0].message.content or ""
    tool_calls_b = response_b.choices[0].message.tool_calls or []
    print(f"Done. finish_reason={response_b.choices[0].finish_reason}, {len(raw_content_b)} chars, {len(tool_calls_b)} tool_calls")

    # -------------------------------------------------------------------------
    # Step 4: Parse both responses
    # -------------------------------------------------------------------------
    separator("Step 4: Parsing responses")
    qwen_parser = Qwen35XMLParser()

    stripped_a = re.sub(r"<think>.*?</think>", "", raw_content_a, flags=re.DOTALL).strip()
    parsed_a = qwen_parser.parse_response(stripped_a)
    warnings_a, _ = check_format_warnings(raw_content_a, tags=["tool_call"], parser_name="qwen35")

    stripped_b = re.sub(r"<think>.*?</think>", "", raw_content_b, flags=re.DOTALL).strip()
    parsed_b = qwen_parser.parse_response(stripped_b)
    warnings_b, _ = check_format_warnings(raw_content_b, tags=["tool_call"], parser_name="qwen35")

    agent_cmd_a = parsed_a.commands[0].arguments.get("command", "") if parsed_a.commands and not parsed_a.commands[0].error else None
    agent_cmd_b = parsed_b.commands[0].arguments.get("command", "") if parsed_b.commands and not parsed_b.commands[0].error else None

    vllm_cmd_b = None
    if tool_calls_b:
        try:
            vllm_cmd_b = json.loads(tool_calls_b[0].function.arguments).get("command", "")
        except (json.JSONDecodeError, IndexError):
            pass

    print(f"Server A / Qwen35XMLParser: {'OK' if not parsed_a.error else parsed_a.error}")
    print(f"Server B / Qwen35XMLParser: {'OK' if not parsed_b.error else parsed_b.error}")
    print(f"Server B / vLLM qwen3_xml:  {'OK' if tool_calls_b else 'no tool_calls'}")

    # -------------------------------------------------------------------------
    # Step 5: Generate PDF
    # -------------------------------------------------------------------------
    pdf_path = args.pdf or "eval/swe_bench/debug_report.pdf"
    separator(f"Step 5: Generating PDF report -> {pdf_path}")
    generate_pdf(
        pdf_path, task, args, messages, response_a, response_b,
        raw_content_a, raw_content_b, tool_calls_b,
        parsed_a, parsed_b, warnings_a, warnings_b,
        agent_cmd_a, agent_cmd_b, vllm_cmd_b,
    )
    print(f"PDF report written to: {pdf_path}")


if __name__ == "__main__":
    main()
