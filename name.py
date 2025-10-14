import sys
from pathlib import Path

IN_PATH = Path("/Users/pyaelinn/tessFinetune/tesstrain/name_corpus/myNER-7tags_ver.1.0.conll")
OUT_PATH = Path("/Users/pyaelinn/tessFinetune/tesstrain/name.txt")


def extract_s_per(src: Path, dst: Path) -> int:
    """Extract only tokens from lines that are tagged 'S-PER' and write tokens to dst.

    CoNLL lines are expected to be whitespace-separated columns, e.g.: 
    <token> <col2> <col3> ... S-PER
    We output only the first column (token), removing columns like 'n' and 'S-PER'.

    Returns the number of tokens written.
    """
    if not src.exists():
        raise FileNotFoundError(f"Input file not found: {src}")

    written = 0
    with src.open("r", encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
        for raw in fin:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            cols = line.split()
            if not cols:
                continue
            if "S-PER" in cols:
                token = cols[0]
                fout.write(token + "\n")
                written += 1
    return written


def main() -> None:
    try:
        count = extract_s_per(IN_PATH, OUT_PATH)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    print(f"Wrote {count} S-PER lines to {OUT_PATH}")


if __name__ == "__main__":
    main()

