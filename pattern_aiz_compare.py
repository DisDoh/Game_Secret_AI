#!/usr/bin/env python3
# pattern_aiz_compare.py
# Analyse deux fichiers .aiz par blocs de 64 bits / 8 octets.
# Usage:
#   python3 pattern_aiz_compare.py disd.jpeg.aiz Flyer_BlueTooth_Poker_8.pdf.aiz
#
# Optionnel: sans arguments, il cherche les deux fichiers du dépôt dans le dossier courant.

from collections import Counter
from pathlib import Path
import math
import sys
import hashlib

DEFAULT_FILES = [
    "disd.jpeg.aiz",
    "Flyer_BlueTooth_Poker_8.pdf.aiz",
]

BLOCK_SIZE = 8  # 64 bits = 8 bytes, d'après Decode_only.py encoding_dim = 64


def entropy(counter, total):
    if total == 0:
        return 0.0
    return -sum((c / total) * math.log2(c / total) for c in counter.values())


def hamming_bytes(a: bytes, b: bytes) -> int:
    return sum((x ^ y).bit_count() for x, y in zip(a, b))


def chunks(data: bytes, block_size: int = BLOCK_SIZE):
    usable = len(data) - (len(data) % block_size)
    return [data[i:i + block_size] for i in range(0, usable, block_size)]


def analyze_file(path: Path):
    data = path.read_bytes()
    ch = chunks(data)

    c = Counter(ch)
    total = len(ch)
    unique = len(c)
    repeated_blocks = sum(1 for block, n in c.items() if n > 1)
    repeated_occurrences = sum(n for block, n in c.items() if n > 1)
    max_block, max_count = c.most_common(1)[0] if c else (b"", 0)

    # Byte-level entropy, useful to see if the .aiz looks random-ish as bytes
    byte_counter = Counter(data)
    byte_entropy = entropy(byte_counter, len(data))

    # Block-level entropy, max would be log2(unique) if perfectly uniform among observed blocks
    block_entropy = entropy(c, total)

    return {
        "path": path,
        "size_bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "num_blocks_64bit": total,
        "trailing_bytes_ignored": len(data) % BLOCK_SIZE,
        "unique_blocks": unique,
        "unique_ratio": unique / total if total else 0,
        "repeated_block_kinds": repeated_blocks,
        "repeated_occurrences": repeated_occurrences,
        "repeat_occurrence_ratio": repeated_occurrences / total if total else 0,
        "most_common_block_hex": max_block.hex(),
        "most_common_block_count": max_count,
        "most_common_block_ratio": max_count / total if total else 0,
        "byte_entropy_bits_per_byte": byte_entropy,
        "block_entropy_bits": block_entropy,
        "top10": c.most_common(10),
        "counter": c,
        "chunks": ch,
    }


def compare(a, b):
    ca = a["counter"]
    cb = b["counter"]

    common_blocks = set(ca) & set(cb)
    common_occurrences_min = sum(min(ca[x], cb[x]) for x in common_blocks)

    # Compare only aligned blocks up to shortest file
    n_aligned = min(len(a["chunks"]), len(b["chunks"]))
    aligned_equal = sum(1 for i in range(n_aligned) if a["chunks"][i] == b["chunks"][i])

    # Bit hamming distance byte-by-byte on common prefix
    da = a["path"].read_bytes()
    db = b["path"].read_bytes()
    prefix = min(len(da), len(db))
    ham = hamming_bytes(da[:prefix], db[:prefix])
    max_bits = prefix * 8

    return {
        "common_unique_blocks": len(common_blocks),
        "common_unique_ratio_vs_a": len(common_blocks) / a["unique_blocks"] if a["unique_blocks"] else 0,
        "common_unique_ratio_vs_b": len(common_blocks) / b["unique_blocks"] if b["unique_blocks"] else 0,
        "common_occurrences_min": common_occurrences_min,
        "common_occurrence_ratio_vs_shorter": common_occurrences_min / min(a["num_blocks_64bit"], b["num_blocks_64bit"]) if min(a["num_blocks_64bit"], b["num_blocks_64bit"]) else 0,
        "aligned_equal_blocks": aligned_equal,
        "aligned_equal_ratio": aligned_equal / n_aligned if n_aligned else 0,
        "hamming_bits_prefix": ham,
        "hamming_ratio_prefix": ham / max_bits if max_bits else 0,
    }


def print_file_report(r):
    print("\n" + "=" * 80)
    print(f"FILE: {r['path']}")
    print("=" * 80)
    print(f"Size:                  {r['size_bytes']:,} bytes")
    print(f"SHA-256:               {r['sha256']}")
    print(f"64-bit blocks:          {r['num_blocks_64bit']:,}")
    print(f"Trailing bytes ignored: {r['trailing_bytes_ignored']}")
    print(f"Unique blocks:          {r['unique_blocks']:,}")
    print(f"Unique ratio:           {r['unique_ratio']:.6%}")
    print(f"Repeated block kinds:   {r['repeated_block_kinds']:,}")
    print(f"Repeated occurrences:   {r['repeated_occurrences']:,}")
    print(f"Repeat ratio:           {r['repeat_occurrence_ratio']:.6%}")
    print(f"Most common block:      {r['most_common_block_hex']}")
    print(f"Most common count:      {r['most_common_block_count']:,}")
    print(f"Most common ratio:      {r['most_common_block_ratio']:.6%}")
    print(f"Byte entropy:           {r['byte_entropy_bits_per_byte']:.4f} / 8 bits")
    print(f"Block entropy:          {r['block_entropy_bits']:.4f} bits")

    print("\nTop 10 repeated 64-bit blocks:")
    for block, count in r["top10"]:
        print(f"  {block.hex()}  x{count}")


def main():
    if len(sys.argv) >= 3:
        paths = [Path(sys.argv[1]), Path(sys.argv[2])]
    else:
        paths = [Path(x) for x in DEFAULT_FILES]

    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        print("Missing files:")
        for m in missing:
            print(" -", m)
        print("\nDownload them from the GitHub repo, then run:")
        print("python3 pattern_aiz_compare.py disd.jpeg.aiz Flyer_BlueTooth_Poker_8.pdf.aiz")
        sys.exit(1)

    r1 = analyze_file(paths[0])
    r2 = analyze_file(paths[1])

    print_file_report(r1)
    print_file_report(r2)

    comp = compare(r1, r2)

    print("\n" + "=" * 80)
    print("COMPARISON")
    print("=" * 80)
    print(f"Common unique 64-bit blocks:               {comp['common_unique_blocks']:,}")
    print(f"Common unique ratio vs file 1:             {comp['common_unique_ratio_vs_a']:.6%}")
    print(f"Common unique ratio vs file 2:             {comp['common_unique_ratio_vs_b']:.6%}")
    print(f"Common occurrences min-count:              {comp['common_occurrences_min']:,}")
    print(f"Common occurrence ratio vs shorter file:   {comp['common_occurrence_ratio_vs_shorter']:.6%}")
    print(f"Aligned equal 64-bit blocks:               {comp['aligned_equal_blocks']:,}")
    print(f"Aligned equal ratio:                       {comp['aligned_equal_ratio']:.6%}")
    print(f"Hamming distance on common byte prefix:    {comp['hamming_bits_prefix']:,} bits")
    print(f"Hamming ratio on common byte prefix:       {comp['hamming_ratio_prefix']:.6%}")

    print("\nInterpretation rapide:")
    if r1["repeat_occurrence_ratio"] > 0.05 or r2["repeat_occurrence_ratio"] > 0.05:
        print("- Beaucoup de répétitions internes: le .aiz révèle des patterns.")
    else:
        print("- Peu de répétitions internes visibles, mais vérifie aussi les blocs communs.")

    if comp["common_unique_blocks"] > 0:
        print("- Il existe des blocs 64-bit identiques entre les deux fichiers.")
        print("  Si le même modèle encode toujours le même octet compressé vers le même bloc,")
        print("  cela confirme un risque de dictionnaire/fréquence.")
    else:
        print("- Aucun bloc 64-bit identique entre les deux fichiers dans cet échantillon.")

    if comp["hamming_ratio_prefix"] < 0.45 or comp["hamming_ratio_prefix"] > 0.55:
        print("- La distance de Hamming s'éloigne de 50%, signe possible de structure non aléatoire.")
    else:
        print("- La distance de Hamming est proche de 50%, ce qui ressemble davantage à du bruit.")


if __name__ == "__main__":
    main()
