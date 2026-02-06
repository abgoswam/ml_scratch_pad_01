"""Reproduce packed-attention logprob mismatch using pure Hugging Face.

This script compares per-sequence logprobs computed in isolation vs. the same
sequences computed inside a flattened packed batch (position_ids reset per segment).
It uses random data and does not depend on any phitrain code.
"""
import argparse
import random
from dataclasses import dataclass
from typing import List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    parser.add_argument("--attn-impl", type=str, default="flash_attention_2")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--vocab-size", type=int, default=None, help="Override vocab size (defaults to model config).")
    parser.add_argument("--pad-id", type=int, default=0)

    parser.add_argument("--max-len", type=int, default=8960, help="Row length after padding.")
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--seqs-per-row", type=int, default=2)
    parser.add_argument("--min-seq-len", type=int, default=3500)
    parser.add_argument("--max-seq-len", type=int, default=5000)

    parser.add_argument("--add-pad-row", action="store_true", help="Append an all-padding row to test length trigger.")
    parser.add_argument("--window-test", action="store_true", help="Test all sliding windows of 4 rows.")
    parser.add_argument("--topk", type=int, default=5)
    return parser.parse_args()


@dataclass
class PackedData:
    input_ids: torch.Tensor            # [1, total_len]
    position_ids: torch.Tensor         # [1, total_len]
    rows_input_ids: torch.Tensor       # [rows, max_len]
    rows_position_ids: torch.Tensor    # [rows, max_len]
    boundaries: List[List[int]]        # per row
    max_len: int
    seq_infos: List[Tuple[int, int, int]]  # (row_idx, start, end)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_rows(
    rows: int,
    seqs_per_row: int,
    max_len: int,
    min_seq_len: int,
    max_seq_len: int,
    vocab_size: int,
    pad_id: int,
    device: str,
) -> Tuple[PackedData, List[List[int]]]:
    """Create packed rows and keep the original sequences."""
    all_rows_ids: List[List[int]] = []
    all_rows_pos: List[List[int]] = []
    boundaries = []
    seq_infos = []
    sequences = []

    for r in range(rows):
        row_ids = []
        row_pos = []
        row_bounds = [0]

        for _ in range(seqs_per_row):
            # Sample a random sequence length
            seq_len = random.randint(min_seq_len, max_seq_len)
            # Ensure we don't overflow the row
            if row_bounds[-1] + seq_len > max_len:
                break
            seq = torch.randint(0, vocab_size, (seq_len,), dtype=torch.long).tolist()
            sequences.append(seq)

            start = row_bounds[-1]
            end = start + seq_len
            seq_infos.append((r, start, end))

            row_ids.extend(seq)
            row_pos.extend(list(range(seq_len)))
            row_bounds.append(end)

        # Pad to max_len
        pad_len = max_len - len(row_ids)
        row_ids.extend([pad_id] * pad_len)
        # Mimic typical packed behavior: reset positions in padding too
        row_pos.extend(list(range(pad_len)))

        # Ensure final boundary reaches max_len
        if row_bounds[-1] < max_len:
            row_bounds.append(max_len)

        all_rows_ids.append(row_ids)
        all_rows_pos.append(row_pos)
        boundaries.append(row_bounds)

    rows_input_ids = torch.tensor(all_rows_ids, dtype=torch.long)
    rows_position_ids = torch.tensor(all_rows_pos, dtype=torch.long)
    input_ids = rows_input_ids.reshape(1, -1).to(device)
    position_ids = rows_position_ids.reshape(1, -1).to(device)

    return PackedData(
        input_ids=input_ids,
        position_ids=position_ids,
        rows_input_ids=rows_input_ids.to(device),
        rows_position_ids=rows_position_ids.to(device),
        boundaries=boundaries,
        max_len=max_len,
        seq_infos=seq_infos,
    ), sequences


def append_pad_row(packed: PackedData, pad_id: int, device: str) -> PackedData:
    """Append an all-padding row to increase total length without adding real tokens."""
    max_len = packed.max_len
    pad_ids = torch.full((1, max_len), pad_id, dtype=torch.long, device=device)
    pad_pos = torch.arange(max_len, dtype=torch.long, device=device).unsqueeze(0)

    rows_input_ids = torch.cat([packed.rows_input_ids, pad_ids], dim=0)
    rows_position_ids = torch.cat([packed.rows_position_ids, pad_pos], dim=0)
    input_ids = rows_input_ids.reshape(1, -1)
    position_ids = rows_position_ids.reshape(1, -1)

    boundaries = packed.boundaries + [[0, 2, max_len]]
    seq_infos = list(packed.seq_infos)

    return PackedData(
        input_ids=input_ids,
        position_ids=position_ids,
        rows_input_ids=rows_input_ids,
        rows_position_ids=rows_position_ids,
        boundaries=boundaries,
        max_len=max_len,
        seq_infos=seq_infos,
    )


def slice_rows(
    packed: PackedData,
    sequences: List[List[int]],
    row_start: int,
    n_rows: int,
) -> Tuple[PackedData, List[List[int]]]:
    """Slice a packed batch to a row window, preserving original sequences."""
    row_end = row_start + n_rows
    rows_input_ids = packed.rows_input_ids[row_start:row_end]
    rows_position_ids = packed.rows_position_ids[row_start:row_end]
    input_ids = rows_input_ids.reshape(1, -1)
    position_ids = rows_position_ids.reshape(1, -1)

    boundaries = [b.copy() for b in packed.boundaries[row_start:row_end]]

    seq_infos = []
    sequences_slice = []
    for idx, (row_idx, start, end) in enumerate(packed.seq_infos):
        if row_start <= row_idx < row_end:
            seq_infos.append((row_idx - row_start, start, end))
            sequences_slice.append(sequences[idx])

    return (
        PackedData(
            input_ids=input_ids,
            position_ids=position_ids,
            rows_input_ids=rows_input_ids,
            rows_position_ids=rows_position_ids,
            boundaries=boundaries,
            max_len=packed.max_len,
            seq_infos=seq_infos,
        ),
        sequences_slice,
    )


def logprobs_from_logits(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    logits, labels = logits[:, :-1], labels[:, 1:]
    valid_mask = labels != ignore_index
    valid_labels = labels.clone()
    valid_labels[~valid_mask] = 0

    if logits.dtype in (torch.float32, torch.float64):
        logits_labels = torch.gather(logits, -1, valid_labels.unsqueeze(-1)).squeeze(-1)
        logsumexp = torch.logsumexp(logits, dim=-1)
        logprobs = logits_labels - logsumexp
    else:
        logprobs_all = torch.nn.functional.log_softmax(logits, dim=-1)
        logprobs = torch.gather(logprobs_all, -1, valid_labels.unsqueeze(-1)).squeeze(-1)

    return logprobs.masked_fill(~valid_mask, 0.0)


def compute_seq_logprobs(model, seq: List[int], device: str, dtype: torch.dtype) -> torch.Tensor:
    input_ids = torch.tensor([seq], dtype=torch.long, device=device)
    pos_ids = torch.arange(len(seq), dtype=torch.long, device=device).unsqueeze(0)
    labels = input_ids.clone()
    with torch.autocast(device_type=device, dtype=dtype, enabled=(dtype != torch.float32)):
        logits = model(input_ids, position_ids=pos_ids, use_cache=False).logits
    return logprobs_from_logits(logits, labels)


def compute_packed_logprobs(model, packed: PackedData, device: str, dtype: torch.dtype) -> torch.Tensor:
    labels = packed.input_ids.clone()
    with torch.autocast(device_type=device, dtype=dtype, enabled=(dtype != torch.float32)):
        logits = model(packed.input_ids, position_ids=packed.position_ids, use_cache=False).logits
    return logprobs_from_logits(logits, labels)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    config = AutoConfig.from_pretrained(args.model)
    vocab_size = args.vocab_size or config.vocab_size

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]

    model_kwargs = {
        "attn_implementation": args.attn_impl,
        "device_map": None,
    }
    if dtype != torch.float32:
        # Newer HF uses `dtype`, older uses `torch_dtype`.
        try:
            model = AutoModelForCausalLM.from_pretrained(
                args.model,
                dtype=dtype,
                **model_kwargs,
            ).to(args.device)
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(
                args.model,
                torch_dtype=dtype,
                **model_kwargs,
            ).to(args.device)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            **model_kwargs,
        ).to(args.device)
    model.eval()

    packed, sequences = make_rows(
        rows=args.rows,
        seqs_per_row=args.seqs_per_row,
        max_len=args.max_len,
        min_seq_len=args.min_seq_len,
        max_seq_len=args.max_seq_len,
        vocab_size=vocab_size,
        pad_id=args.pad_id,
        device=args.device,
    )

    def run_case(label: str, packed_case: PackedData, sequences_case: List[List[int]]):
        packed_logprobs = compute_packed_logprobs(model, packed_case, args.device, dtype)

        max_diff = 0.0
        avg_sum = 0.0
        count = 0
        top_entries = []

        for idx, (row_idx, start, end) in enumerate(packed_case.seq_infos):
            flat_offset = row_idx * packed_case.max_len
            packed_slice = packed_logprobs[0, flat_offset + start : flat_offset + end - 1].cpu()

            seq = sequences_case[idx]
            ind_logprobs = compute_seq_logprobs(model, seq, args.device, dtype)[0].cpu()

            diff = (packed_slice - ind_logprobs).abs()
            avg = diff.mean().item()
            mx = diff.max().item()
            max_diff = max(max_diff, mx)
            avg_sum += avg
            count += 1

            top_entries.append((mx, idx, row_idx, start, end))

        avg_diff = avg_sum / max(1, count)
        top_entries.sort(reverse=True, key=lambda x: x[0])

        print(f"{label}: avg_diff={avg_diff:.6f}, max_diff={max_diff:.6f}")
        for mx, idx, row_idx, start, end in top_entries[: args.topk]:
            print(f"  seq {idx} row {row_idx} [{start},{end}) max_diff={mx:.6f}")

    print(f"Model: {args.model}")
    print(f"attn_impl: {args.attn_impl}, dtype: {args.dtype}")
    print(f"rows={args.rows}, max_len={args.max_len}, total_len={args.rows * args.max_len}")

    run_case("base", packed, sequences)

    if args.add_pad_row:
        packed_plus = append_pad_row(packed, args.pad_id, args.device)
        # sequences unchanged
        run_case("base + pad_row", packed_plus, sequences)

    if args.window_test and args.rows >= 4:
        print("\nSliding 4-row windows (sliced from base):")
        for start in range(0, args.rows - 3):
            packed_window, sequences_window = slice_rows(packed, sequences, start, 4)
            run_case(f"rows {start}-{start+3}", packed_window, sequences_window)


if __name__ == "__main__":
    main()
