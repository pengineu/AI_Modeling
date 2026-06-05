"""Score a Task3 output txt by fruit/style label match (precision@K).

Each row of the output corresponds to a query image in NUMERIC id order
(row 0 -> 0.jpg, row 1 -> 1.jpg, ...). For every retrieved id we look up its
TRUE (style, fruit) from the labels CSV and compare to the query's TRUE label.
This is exactly the metric the TA grades on (fruit & style label match).

Usage:
    python -m src.task3.score_output --txt release/202502204.test.task3.txt \
        --labels data/train/train_labels.csv
"""

import argparse
import csv
import re
from collections import defaultdict

from src.common.labels import FRUIT_ID2NAME, STYLE_ID2NAME

TUP = re.compile(r"\(([^,()]+),\s*(\d+),\s*(\d+)\)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--txt", required=True)
    ap.add_argument("--labels", required=True, help="CSV with file_name,style,fruit")
    args = ap.parse_args()

    lab = {}
    with open(args.labels, newline="") as f:
        for row in csv.DictReader(f):
            lab[row["file_name"].strip()] = (int(row["style"]), int(row["fruit"]))

    rows = []
    with open(args.txt, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append([(m[0].strip(), int(m[1]), int(m[2])) for m in TUP.findall(line)])

    # row i  ->  query id = i-th image in numeric order
    query_ids = sorted(lab, key=lambda x: (0, int(x.split(".")[0]))
                       if x.split(".")[0].isdigit() else (1, x))
    if len(query_ids) != len(rows):
        print(f"[warn] rows={len(rows)} but labels CSV has {len(query_ids)} ids; "
              "assuming row i == i-th sorted id")

    fruit_hit = style_hit = both_hit = tot = 0
    pf = defaultdict(lambda: [0, 0]); ps = defaultdict(lambda: [0, 0])
    skipped = 0
    for i, r in enumerate(rows):
        qid = query_ids[i] if i < len(query_ids) else None
        if qid not in lab:
            skipped += 1; continue
        sq, fq = lab[qid]
        for rid, _, _ in r:
            if rid not in lab:
                skipped += 1; continue
            sr, fr = lab[rid]
            tot += 1
            fm = fr == fq; sm = sr == sq
            fruit_hit += fm; style_hit += sm; both_hit += fm and sm
            pf[fq][0] += fm; pf[fq][1] += 1
            ps[sq][0] += sm; ps[sq][1] += 1

    K = len(rows[0]) if rows else 0
    print(f"queries(rows): {len(rows)} | TOP_K: {K} | scored slots: {tot} | skipped: {skipped}")
    print(f"\n=== label match precision@{K} ===")
    print(f"  fruit : {fruit_hit/tot:.4f}")
    print(f"  style : {style_hit/tot:.4f}")
    print(f"  both  : {both_hit/tot:.4f}   (fruit AND style)")
    print(f"  (random baseline: fruit 0.167 / style 0.333 / both 0.056)")
    print("\nper-class fruit P@K:",
          {FRUIT_ID2NAME[k]: round(v[0]/v[1], 3) for k, v in sorted(pf.items())})
    print("per-class style P@K:",
          {STYLE_ID2NAME[k]: round(v[0]/v[1], 3) for k, v in sorted(ps.items())})


if __name__ == "__main__":
    main()
