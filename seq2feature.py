#!/usr/bin/env python3
# coding: utf-8
"""
Convert DNA sequences to dot-matrix / one-hot features
and save them as a compressed NumPy file.

Example
-------
python seq2feature.py \
    --fasta data/train.fasta \
    --kmer 3 \
    --base-height 3 \
    --feature both \                    #["dot", "one-hot", "both"]
    --out-prefix data/train_features
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as dt
from pathlib import Path
from typing import List, Tuple

import numpy as np
from scipy.sparse import csr_matrix
import os

# ─────────────────────────────────────────────────────────────
#  Feature builders
# ─────────────────────────────────────────────────────────────
def to_dot_matrix(seq1: str, seq2: str, consecutive_length: int = 3) -> np.ndarray:
    if consecutive_length > min(len(seq1), len(seq2)):
        raise ValueError("k-mer longer than sequence length")

    n1 = len(seq1) - consecutive_length + 1
    n2 = len(seq2) - consecutive_length + 1
    mat = np.zeros((n1, n2), dtype=np.float32)

    for i in range(n1):
        s1 = seq1[i : i + consecutive_length]
        for j in range(n2):
            if s1 == seq2[j : j + consecutive_length]:
                mat[i, j] = 1.0

    padded = np.zeros((len(seq1), len(seq1)), dtype=np.float32)
    padded[:n1, :n2] = mat
    return padded


def seq_to_one_hot(seq: str, base_height: int, pad: bool = True) -> np.ndarray:
    """Return (4*base_height)×len(seq) one-hot matrix.
       If pad=True, left-top pad into len(seq)×len(seq) square."""
    mapping = {"A": 0, "C": 1, "G": 2, "T": 3}
    seq = seq.upper()

    one_hot = np.zeros((4 * base_height, len(seq)), dtype=np.float32)
    for i, base in enumerate(seq):
        if base in mapping:
            idx = mapping[base] * base_height
            one_hot[idx : idx + base_height, i] = 1.0

    if not pad:                       # ← 只有 one-hot 模式才走這裡
        return one_hot

    padded = np.zeros((len(seq), len(seq)), dtype=np.float32)
    padded[: one_hot.shape[0], : one_hot.shape[1]] = one_hot
    return padded


def complementary_dna(seq: str) -> str:
    return seq.translate(str.maketrans("ACGTacgt", "TGCAtgca"))


def process_sequence(
    seq: str,
    kmer: int,
    base_height: int,
    feature_type: str,
) -> Tuple[csr_matrix, ...]:
    rev_seq = seq[::-1]
    rev_comp_seq = complementary_dna(rev_seq)

    mats: List[csr_matrix] = []

    if feature_type in ("dot", "both"):
        dot_self = csr_matrix(to_dot_matrix(seq, seq, kmer))
        dot_rev = csr_matrix(np.fliplr(to_dot_matrix(seq, rev_seq, kmer)))
        dot_rev_comp = csr_matrix(np.fliplr(to_dot_matrix(seq, rev_comp_seq, kmer)))
        mats.extend([dot_self, dot_rev, dot_rev_comp])

    if feature_type in ("one-hot", "both"):
        # pad除非「只做 one-hot」
        pad_one_hot = feature_type != "one-hot"
        one_hot = csr_matrix(seq_to_one_hot(seq, base_height, pad=pad_one_hot))
        mats.append(one_hot)

    return tuple(mats)


# ─────────────────────────────────────────────────────────────
#  I/O helpers
# ─────────────────────────────────────────────────────────────
def parse_fasta(path: Path) -> List[str]:
    sequences: List[str] = []
    with path.open() as fh:
        seq_chunks: List[str] = []
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if seq_chunks:
                    sequences.append("".join(seq_chunks))
                    seq_chunks.clear()
            else:
                seq_chunks.append(line)
        if seq_chunks:
            sequences.append("".join(seq_chunks))
    if not sequences:
        raise ValueError(f"No sequences found in {path}")
    return sequences


# ─────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────
def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert FASTA sequences to sparse feature matrices"
    )
    p.add_argument("--fasta", required=True, type=Path, help="Input FASTA file")
    p.add_argument("--kmer", type=int, default=3, help="Consecutive match length")
    p.add_argument(
        "--base-height",
        type=int,
        default=1,
        help="Rows per nucleotide in one-hot",
    )
    p.add_argument(
        "--feature",
        choices=["dot", "one-hot", "both"],
        default="both",
        help="Which feature(s) to compute",
    )
    p.add_argument(
        "--out-prefix",
        type=Path,
        default=Path("features"),
        help="Output prefix (no extension)",
    )
    p.add_argument(
        "--n-proc", "--max-workers",
        dest="n_proc",
        type=int,
        default=os.cpu_count(),
        help="ProcessPool max workers (default: CPU count)",
    )
    return p.parse_args()


# ─────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────
def main() -> None:
    args = get_args()

    print("─" * 60)
    print(f"Started  : {dt.datetime.now().isoformat(timespec='seconds')}")
    print(f"Reading  : {args.fasta}")
    sequences = parse_fasta(args.fasta)

    if args.feature in ("dot", "both"):
        seq_len_set = {len(s) for s in sequences}
        if len(seq_len_set) != 1:
            raise ValueError(
                "All sequences must have the same length when using "
                "'dot' or 'both' features (required for square padding)."
            )
        seq_len = seq_len_set.pop()
        print(f"Seq count: {len(sequences)} (length={seq_len})")
    else:
        print(f"Seq count: {len(sequences)})")

    print(f"Feature  : {args.feature}")
    print("Building features ...")

    with cf.ProcessPoolExecutor(max_workers=args.n_proc) as exe:
        feats = list(
            exe.map(
                process_sequence,
                sequences,
                [args.kmer] * len(sequences),
                [args.base_height] * len(sequences),
                [args.feature] * len(sequences),
            )
        )

    feature_array = np.stack([np.array(mats, dtype=object) for mats in feats], axis=0,)
    feature_array = feature_array[np.newaxis, ...]
    out_file = args.out_prefix.with_suffix(".npz")
    np.savez_compressed(out_file, nested_structure=feature_array)

    print(f"Saved to : {out_file}")
    print(f"Finished : {dt.datetime.now().isoformat(timespec='seconds')}")
    print("─" * 60)


if __name__ == "__main__":
    main()
