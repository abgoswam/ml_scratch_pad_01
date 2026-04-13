# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Reconstruct the exact model input for the SWE-bench eval pipeline as run by run_eval.sh.

Mimics the harbor -> mini-swe-agent -> LitellmModel -> vLLM flow:
  1. Harbor runs: mini-swe-agent --yolo --model=hosted_vllm/vllm_hosted_model --task=...
  2. mini-swe-agent loads mini.yaml (default config)
  3. DefaultAgent renders system_template and instance_template with Jinja2
  4. LitellmModel calls litellm.completion(messages=..., tools=[BASH_TOOL])
  5. vLLM receives the request, applies the Qwen3 chat template with tools=
  6. The rendered prompt is what the model actually sees

This script reconstructs steps 2-5 locally and generates a PDF report.
"""

import argparse
import json
import platform
import re
import sys
import textwrap
from datetime import datetime
from pathlib import Path

import yaml
from datasets import load_dataset
from jinja2 import StrictUndefined, Template
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

# Paths
MODEL_PATH = str(PROJECT_ROOT / "_ckpts" / "Qwen3-14B-modified")
TRAIN_CONFIG_PATH = str(PROJECT_ROOT / "phitrain" / "recipes" / "swe_agent" / "configs" / "swebench.yaml")
MINI_SWE_AGENT_DIR = PROJECT_ROOT / "mini-swe-agent" / "src" / "minisweagent"
MINI_YAML_PATH = MINI_SWE_AGENT_DIR / "config" / "mini.yaml"

# mini-swe-agent's BASH_TOOL (from models/utils/actions_toolcall.py)
BASH_TOOL = {
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


def load_mini_yaml() -> dict:
    with open(MINI_YAML_PATH) as f:
        return yaml.safe_load(f)


def render_template(template_str: str, **kwargs) -> str:
    return Template(template_str, undefined=StrictUndefined).render(**kwargs)


def build_harbor_eval_flow(tokenizer, task_text: str) -> dict:
    """Reconstruct the exact harbor -> mini-swe-agent -> vLLM flow."""
    config = load_mini_yaml()
    agent_config = config.get("agent", {})
    model_config = config.get("model", {})
    env_config = config.get("environment", {})

    system_template = agent_config.get("system_template", "")
    instance_template = agent_config.get("instance_template", "")

    # Template variables (merged from agent, model, environment, platform)
    uname = platform.uname()._asdict()
    template_vars = {
        **agent_config,
        **model_config,
        **env_config,
        **uname,
        "task": task_text,
        "n_model_calls": 0,
        "model_cost": 0.0,
    }

    # Step 1: DefaultAgent renders templates
    system_content = render_template(system_template, **template_vars)
    user_content = render_template(instance_template, **template_vars)

    # Step 2: LitellmModel._prepare_messages_for_api builds messages
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]

    # Step 3: LitellmModel._query calls litellm.completion(messages=..., tools=[BASH_TOOL])
    # Step 4: litellm forwards to vLLM which applies the chat template
    rendered_prompt = tokenizer.apply_chat_template(
        messages,
        tools=[BASH_TOOL],
        tokenize=False,
        add_generation_prompt=True,
    )

    return {
        "config_file": str(MINI_YAML_PATH),
        "system_template": system_template,
        "instance_template": instance_template,
        "system_content": system_content,
        "user_content": user_content,
        "messages": messages,
        "tools": [BASH_TOOL],
        "rendered_prompt": rendered_prompt,
        "observation_template": model_config.get("observation_template", ""),
        "format_error_template": model_config.get("format_error_template", ""),
    }


def build_training_flow(tokenizer, task_text: str) -> dict:
    """Reconstruct the training pipeline flow."""
    formatter = create_miniswe_message_formatter(TRAIN_CONFIG_PATH, tokenizer=tokenizer)
    messages = formatter(task_text)

    rendered_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    return {
        "config_file": TRAIN_CONFIG_PATH,
        "system_content": messages[0]["content"],
        "user_content": messages[1]["content"],
        "messages": messages,
        "tools": None,
        "rendered_prompt": rendered_prompt,
    }


# ---------------------------------------------------------------------------
# PDF helpers
# ---------------------------------------------------------------------------

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


def _code_block(text: str, max_width: int = 120) -> Preformatted:
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


def _section_header(text: str) -> Paragraph:
    style = ParagraphStyle("H1", fontName="Helvetica-Bold", fontSize=13,
                            spaceBefore=14, spaceAfter=6,
                            textColor=colors.HexColor("#2c3e50"))
    return Paragraph(text, style)


def _subsection_header(text: str) -> Paragraph:
    style = ParagraphStyle("H2", fontName="Helvetica-Bold", fontSize=10,
                            spaceBefore=8, spaceAfter=4,
                            textColor=colors.HexColor("#34495e"))
    return Paragraph(text, style)


def _note(text: str) -> Paragraph:
    style = ParagraphStyle("Note", fontName="Helvetica", fontSize=8, leading=10,
                            textColor=colors.HexColor("#7f8c8d"))
    return Paragraph(text, style)


def _body(text: str) -> Paragraph:
    style = ParagraphStyle("Body", fontName="Helvetica", fontSize=8, leading=10)
    return Paragraph(text, style)


def _esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def generate_pdf(pdf_path: str, task, eval_flow, train_flow, tokenizer) -> None:
    doc = SimpleDocTemplate(
        pdf_path, pagesize=letter,
        leftMargin=0.4 * inch, rightMargin=0.4 * inch,
        topMargin=0.4 * inch, bottomMargin=0.4 * inch,
    )
    page_w = letter[0] - 0.8 * inch
    col_w = (page_w - 0.35 * inch) / 2

    title_style = ParagraphStyle("Title2", fontName="Helvetica-Bold", fontSize=16,
                                  spaceAfter=6, alignment=1)

    elements = []

    # ── Title ──
    elements.append(Paragraph("Harbor Eval Flow: Model Input Reconstruction", title_style))
    elements.append(_note(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"))
    elements.append(Spacer(1, 12))

    # ── Section 1: What run_eval.sh does ──
    elements.append(_section_header("1. What run_eval.sh Does"))
    flow_data = [
        ["Step", "Component", "Action"],
        ["1", "run_eval.sh", "Starts vLLM with --enable-auto-tool-choice\n--tool-call-parser qwen3_xml"],
        ["2", "run_eval.sh", "Runs: harbor run --config config_qwen3-14b.yaml"],
        ["3", "Harbor", "Spins up Docker container per task,\ninstalls mini-swe-agent via uv"],
        ["4", "Harbor", "Runs inside container:\nmini-swe-agent --yolo\n  --model=hosted_vllm/vllm_hosted_model\n  --task=<problem_statement>"],
        ["5", "mini-swe-agent", "Loads mini.yaml (default config)\nNo -c flag passed, so uses builtin default\n(mini.py:63)"],
        ["6", "mini-swe-agent\nDefaultAgent", "Renders system_template and\ninstance_template with Jinja2"],
        ["7", "mini-swe-agent\nLitellmModel", "Calls litellm.completion(\n  messages=[system, user],\n  tools=[BASH_TOOL])\n(litellm_model.py:65-70)"],
        ["8", "litellm", "Forwards to vLLM OpenAI-compatible API"],
        ["9", "vLLM", "Applies Qwen3 chat template with tools=\nRenders tool schemas as XML in system block"],
        ["10", "Model", "Generates response with <tool_call> XML"],
        ["11", "vLLM", "qwen3_xml parser strips <tool_call> XML\nfrom content, populates tool_calls field"],
        ["12", "mini-swe-agent\nLitellmModel", "Reads response.choices[0].message.tool_calls\nExtracts bash command\n(litellm_model.py:117)"],
    ]
    elements.append(_make_table(flow_data, [0.4 * inch, 1.3 * inch, page_w - 1.7 * inch]))
    elements.append(Spacer(1, 12))

    # ── Section 2: Config ──
    elements.append(_section_header("2. Configuration"))
    elements.append(_subsection_header("2a. Harbor config (config_qwen3-14b.yaml)"))
    elements.append(_code_block(
        "n_attempts: 1\n"
        "orchestrator:\n"
        "  n_concurrent_trials: 16\n"
        "environment:\n"
        "  type: docker\n"
        "  delete: true\n"
        "verifier:\n"
        "  override_timeout_sec: 300\n"
        "agents:\n"
        "  - name: mini-swe-agent\n"
        "    model_name: hosted_vllm/vllm_hosted_model\n"
        "    override_timeout_sec: 1800"
    ))
    elements.append(Spacer(1, 6))

    elements.append(_subsection_header("2b. mini-swe-agent config (mini.yaml — loaded by default)"))
    elements.append(_note("Source: mini-swe-agent/src/minisweagent/config/mini.yaml"))
    elements.append(_code_block(
        f"system_template: |\n"
        + textwrap.indent(eval_flow["system_template"], "  ")
        + f"\ninstance_template: |\n"
        + textwrap.indent(eval_flow["instance_template"][:500] + "\n  ...", "  ")
    ))
    elements.append(Spacer(1, 6))

    elements.append(_subsection_header("2c. BASH_TOOL schema (passed via tools= in API call)"))
    elements.append(_code_block(json.dumps(BASH_TOOL, indent=2)))
    elements.append(Spacer(1, 12))

    # ── Section 3: Messages before chat template ──
    elements.append(PageBreak())
    elements.append(_section_header("3. Messages Before Chat Template"))
    elements.append(_note(
        "These are the messages that LitellmModel passes to litellm.completion(). "
        "The system_template and instance_template have been rendered with Jinja2."
    ))
    elements.append(Spacer(1, 6))

    elements.append(_subsection_header("3a. System message"))
    elements.append(_code_block(eval_flow["system_content"]))
    elements.append(Spacer(1, 6))

    elements.append(_subsection_header("3b. User message"))
    elements.append(_code_block(eval_flow["user_content"]))
    elements.append(Spacer(1, 12))

    # ── Section 4: Final rendered prompt ──
    elements.append(PageBreak())
    elements.append(_section_header("4. Final Rendered Prompt (what the model sees)"))
    elements.append(_note(
        "This is tokenizer.apply_chat_template(messages, tools=[BASH_TOOL]). "
        "vLLM runs this internally when it receives the API request. "
        "The Qwen3 chat template renders tool schemas as XML in the system block, "
        "followed by the original system message content, then the user message."
    ))
    elements.append(Spacer(1, 6))

    # Annotate sections of the rendered prompt
    prompt = eval_flow["rendered_prompt"]
    elements.append(_code_block(prompt))
    elements.append(Spacer(1, 8))

    # Token count
    eval_tokens = tokenizer.encode(prompt)
    elements.append(_body(f"<b>Total: {len(prompt)} chars, {len(eval_tokens)} tokens</b>"))
    elements.append(Spacer(1, 12))

    # ── Section 5: Anatomy of the rendered prompt ──
    elements.append(PageBreak())
    elements.append(_section_header("5. Anatomy of the Rendered Prompt"))
    elements.append(_note("Breaking down where each part of the final prompt comes from."))
    elements.append(Spacer(1, 6))

    # Parse the prompt into sections
    sections = []
    # Tool section from chat template
    tool_match = re.search(r"(# Tools.*?</IMPORTANT>)", prompt, re.DOTALL)
    if tool_match:
        sections.append(("Chat template (Qwen3)\ntokenizer_config.json",
                         "Tool schemas + XML format instructions",
                         tool_match.group(1)[:300] + "..."))

    # System message content (after tool section)
    sys_match = re.search(r"</IMPORTANT>\n\n(.*?)<\|im_end\|>", prompt, re.DOTALL)
    if sys_match:
        sections.append(("mini.yaml\nsystem_template",
                         "Agent role description",
                         sys_match.group(1).strip()[:300]))

    # User message
    user_match = re.search(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", prompt, re.DOTALL)
    if user_match:
        sections.append(("mini.yaml\ninstance_template",
                         "Task + workflow instructions",
                         user_match.group(1).strip()[:300] + "..."))

    # Generation prompt
    if "<|im_start|>assistant" in prompt:
        sections.append(("Chat template (Qwen3)",
                         "Generation prompt",
                         "<|im_start|>assistant\\n<think>\\n"))

    anatomy_data = [["Source", "Purpose", "Content (truncated)"]]
    for source, purpose, content in sections:
        anatomy_data.append([source, purpose, content])
    elements.append(_make_table(anatomy_data, [1.5 * inch, 1.3 * inch, page_w - 2.8 * inch]))
    elements.append(Spacer(1, 12))

    # ── Section 6: Eval vs Training comparison ──
    elements.append(PageBreak())
    elements.append(_section_header("6. Eval vs Training: Pipeline Comparison"))

    cmp_data = [
        ["", "Eval (harbor / mini-swe-agent)", "Training (phitrain)"],
        ["Config", "mini.yaml", "swebench.yaml"],
        ["System message", "Clean: 'You are a helpful\nassistant that can interact\nwith a computer.'",
         "Clean: 'You are a helpful assistant\nthat can interact multiple times\nwith a computer shell...'"],
        ["Tool schemas\nin system msg?", "NO — injected by vLLM\nvia chat template when\nit sees tools=",
         "YES — baked in by formatter\nvia apply_chat_template(tools=)\nat data prep time"],
        ["tools= in API", "YES — tools=[BASH_TOOL]\n(litellm_model.py:68)", "NO"],
        ["Tool format\ninstructions", "From chat template\n(Qwen3 tokenizer_config.json)",
         "From chat template\n(same — baked into system msg)"],
        ["Response parsing", "vLLM qwen3_xml parser\n-> tool_calls field", "Qwen35XMLParser\non raw content"],
        ["Format conflict?", "NO — mini.yaml has no\ncompeting format instructions",
         "NO — swebench.yaml has no\ncompeting format instructions"],
    ]
    elements.append(_make_table(cmp_data, [1.5 * inch, (page_w - 1.5 * inch) / 2, (page_w - 1.5 * inch) / 2]))
    elements.append(Spacer(1, 12))

    # ── Section 7: Side-by-side rendered prompts ──
    elements.append(PageBreak())
    elements.append(_section_header("7. Side-by-Side: Final Rendered Prompts"))
    elements.append(_note("Red rows show where the eval and training prompts diverge."))
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
        wrap_lines(eval_flow["rendered_prompt"]),
        wrap_lines(train_flow["rendered_prompt"]),
        "Eval (harbor/mini-swe-agent)",
        "Training (phitrain)",
        col_w,
    ))
    elements.append(Spacer(1, 8))

    # Token comparison
    train_tokens = tokenizer.encode(train_flow["rendered_prompt"])
    tok_data = [
        ["", "Eval", "Training", "Diff"],
        ["Chars", str(len(eval_flow["rendered_prompt"])),
         str(len(train_flow["rendered_prompt"])),
         str(len(eval_flow["rendered_prompt"]) - len(train_flow["rendered_prompt"]))],
        ["Tokens", str(len(eval_tokens)), str(len(train_tokens)),
         str(len(eval_tokens) - len(train_tokens))],
    ]
    elements.append(_make_table(tok_data, [1.5 * inch, 1.5 * inch, 1.5 * inch, 1.5 * inch]))
    elements.append(Spacer(1, 12))

    # ── Section 8: Key findings ──
    elements.append(_section_header("8. Key Findings"))

    # Check if tool schema sections match
    tool_re = re.compile(r"# Tools.*?</IMPORTANT>", re.DOTALL)
    eval_tool = tool_re.search(eval_flow["rendered_prompt"])
    train_tool = tool_re.search(train_flow["rendered_prompt"])

    findings = []
    if eval_tool and train_tool:
        tools_match = eval_tool.group(0).strip() == train_tool.group(0).strip()
        findings.append(f"Tool schema + format instructions: <b>{'IDENTICAL' if tools_match else 'DIFFERENT'}</b>")
    findings.append(
        "Both pipelines use the same Qwen3 chat template to render tool schemas. "
        "The XML format instructions are identical regardless of injection method."
    )
    findings.append(
        "The difference is in the system/user message content — different templates "
        "produce different task instructions, workflow guidance, and submission rules."
    )
    findings.append(
        "Neither pipeline has conflicting format instructions. mini.yaml's system_template "
        "is clean (no mswea_bash_command), and phitrain's swebench.yaml is also clean."
    )

    for i, f in enumerate(findings, 1):
        elements.append(_body(f"{i}. {f}"))
        elements.append(Spacer(1, 4))

    doc.build(elements)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect harbor eval flow model input")
    parser.add_argument("--instance-id", type=str, default=None)
    parser.add_argument("--model-path", type=str, default=MODEL_PATH)
    parser.add_argument("--pdf", type=str, default="eval/swe_bench/harbor_flow_report.pdf")
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

    print("Building eval flow (harbor -> mini-swe-agent -> vLLM)...")
    eval_flow = build_harbor_eval_flow(tokenizer, task["problem_statement"])

    print("Building training flow (phitrain)...")
    train_flow = build_training_flow(tokenizer, task["problem_statement"])

    # Write rendered prompts to text files
    output_dir = Path(args.pdf).parent
    eval_txt = output_dir / "model_turn_eval.txt"
    train_txt = output_dir / "model_turn_train.txt"

    eval_txt.write_text(eval_flow["rendered_prompt"])
    print(f"Wrote: {eval_txt}")

    train_txt.write_text(train_flow["rendered_prompt"])
    print(f"Wrote: {train_txt}")

    print(f"\nGenerating PDF -> {args.pdf}")
    generate_pdf(args.pdf, task, eval_flow, train_flow, tokenizer)
    print(f"Done: {args.pdf}")


if __name__ == "__main__":
    main()
