#!/usr/bin/env python3
"""Generate a SYNTHETIC .cin (random letter codes) covering the most frequent
characters of a sentence file. For evaluating the pipeline end to end only;
it has no relation to any real Boshiamy / 嘸蝦米 table and must not be shipped.
Users evaluate with their own table via `boshiamy-eval --cin`."""
import argparse, collections, random, string

ap = argparse.ArgumentParser()
ap.add_argument("--sentences", required=True)
ap.add_argument("--output", required=True)
ap.add_argument("--top", type=int, default=6000)
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--max-lines", type=int, default=2_000_000)
a = ap.parse_args()
random.seed(a.seed)
cnt = collections.Counter()
with open(a.sentences, encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= a.max_lines:
            break
        cnt.update(ch for ch in line.strip() if "一" <= ch <= "鿿")
chars = [c for c, _ in cnt.most_common(a.top)]
letters = string.ascii_lowercase
used = {}
with open(a.output, "w", encoding="utf-8") as out:
    out.write("# SYNTHETIC random-code table for pipeline evaluation only. Not a real input-method table.\n")
    out.write("%ename SyntheticRandom\n%cname 合成隨機碼表\n%chardef begin\n")
    for rank, ch in enumerate(chars):
        # Frequent chars get shorter codes, like a real table; allow a few collisions.
        length = 2 if rank < 400 else 3 if rank < 4000 else 4
        for _ in range(100):
            code = "".join(random.choice(letters) for _ in range(length))
            if used.get(code, 0) < 2:
                used[code] = used.get(code, 0) + 1
                break
        out.write(f"{code}\t{ch}\n")
    out.write("%chardef end\n")
print(f"wrote {len(chars)} chars, {len(used)} distinct codes")
