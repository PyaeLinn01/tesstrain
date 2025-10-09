import json
from pathlib import Path

SRC = Path("/Users/pyaelinn/tessFinetune/tesstrain/nrc_draft.json")
DST = Path("/Users/pyaelinn/tessFinetune/tesstrain/nrc.json")

MYANMAR_DIGITS = {"0":"၀","1":"၁","2":"၂","3":"၃","4":"၄","5":"၅","6":"၆","7":"၇","8":"၈","9":"၉","10":"၁၀","11":"၁၁","12":"၁၂","13":"၁၃","14":"၁၄"}

def to_myanmar_numerals(s: str) -> str:
    return "".join(MYANMAR_DIGITS.get(ch, ch) for ch in s)

def main():
    data = json.loads(SRC.read_text(encoding="utf-8"))
    # Output structure: { "1": [ {my,en}, ... ], "2": [ ... ], ... }
    out = {}
    # Suffix variants: Myanmar -> English
    variants = [
        ("နိုင်", "C"),
        ("ဧည့်", "AC"),
        ("ပြု", "NC"),
        ("စ", "V"),
        ("သ", "M"),
        ("သီ", "N"),
    ]
    for region in data:
        rid = str(region.get("id", ""))
        my_num = to_myanmar_numerals(rid)
        codes = []
        for dist in region.get("districts", []):
            for tw in dist.get("townships", []):
                code = tw.get("code", {})
                my = code.get("my")
                en = code.get("en")
                if not my or not en:
                    continue
                for my_sfx, en_sfx in variants:
                    my_form = f"{my_num}/{my}({my_sfx})"
                    en_form = f"{en} ({en_sfx})"
                    codes.append({"my": my_form, "en": en_form})
        if rid in out:
            out[rid].extend(codes)
        else:
            out[rid] = codes
    DST.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()

