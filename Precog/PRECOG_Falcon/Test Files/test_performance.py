
# Measure speedup from cached sequential to cached multi-token processing.

import csv
import statistics
import time
from pathlib import Path

import torch
from test_correctness import PRECOG, SCENARIOS, COUNTS, match

RESULTS = Path(__file__).resolve().parent / "results"
REPEATS = 3


def timed(fn):
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = fn()
    torch.cuda.synchronize()
    return result, time.perf_counter() - start


def median_time(fn):
    timed(fn)
    return statistics.median(
        timed(fn)[1] for _ in range(REPEATS)
    )


def main():
    RESULTS.mkdir(exist_ok=True)
    model = PRECOG()
    rows = []

    for topic, document, question in SCENARIOS:
        length, state = model.prepare(document)
        question_ids = model.tokens(question * 4)

        for count in COUNTS:
            args = (state, length, question_ids[:count])

            sequential = lambda: model.continue_tokens(
                *args, method="sequential"
            )
            multitoken = lambda: model.continue_tokens(
                *args, method="multitoken"
            )

            matched, total = match(
                sequential()[0], multitoken()[0]
            )

            seq_s = median_time(sequential)
            multi_s = median_time(multitoken)

            rows.append({
                "topic": topic,
                "context_tokens": length,
                "tokens": count,
                "matched": matched,
                "total": total,
                "sequential_s": seq_s,
                "multitoken_s": multi_s,
                "speedup": seq_s / multi_s,
            })

            print(
                f"{topic} {count} tokens | {matched}/{total} | "
                f"{seq_s:.3f}s vs {multi_s:.3f}s | "
                f"{seq_s / multi_s:.2f}x"
            )

    path = RESULTS / "cached_performance.csv"

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    median = statistics.median(r["speedup"] for r in rows)
    print(f"Median speedup: {median:.2f}x")
    print("Saved:", path)


if __name__ == "__main__":
    main()
