
import csv
import statistics
import time
from pathlib import Path

import torch

from test_correctness import (
    PRECOG,
    SCENARIOS,
    TOKEN_COUNTS,
    agreement,
)


REPEATS = 3
OUTPUT_FILE = Path(__file__).resolve().parent / "performance_results.csv"


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def run(precog, state, position, ids, method):
    sync()
    start = time.perf_counter()

    result = precog.continue_tokens(
        state, position, ids, method=method
    )

    sync()
    return result, time.perf_counter() - start


def measure(precog, state, position, ids, method):
    run(precog, state, position, ids, method)

    times = [
        run(precog, state, position, ids, method)[1]
        for _ in range(REPEATS)
    ]

    return statistics.median(times)


def main():
    print("Falcon Mamba PRECOG performance tests")

    precog = PRECOG()
    results = []

    for name, document, question in SCENARIOS:
        context_length, state = precog.prepare(document)
        question_ids = precog.tokens(question * 4)

        print(f"\n{name}: {context_length} context tokens")

        for count in TOKEN_COUNTS:
            ids = question_ids[:count]

            ref, _ = run(
                precog, state, context_length, ids, "sequential"
            )
            new, _ = run(
                precog, state, context_length, ids, "multitoken"
            )

            matched, total = agreement(ref[0], new[0])

            sequential = measure(
                precog, state, context_length, ids, "sequential"
            )
            optimized = measure(
                precog, state, context_length, ids, "multitoken"
            )
            speedup = sequential / optimized

            results.append({
                "scenario": name,
                "context_tokens": context_length,
                "continuation_tokens": count,
                "matched": matched,
                "total": total,
                "sequential_seconds": round(sequential, 4),
                "multitoken_seconds": round(optimized, 4),
                "speedup": round(speedup, 3),
            })

            print(
                f"{count:2} tokens | Match {matched}/{total} | "
                f"Sequential {sequential:.4f}s | "
                f"Multi {optimized:.4f}s | "
                f"Speedup {speedup:.2f}x",
                flush=True,
            )

    with OUTPUT_FILE.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=list(results[0])
        )
        writer.writeheader()
        writer.writerows(results)

    median = statistics.median(
        row["speedup"] for row in results
    )

    print(f"\nMedian speedup: {median:.2f}x")
    print("Results saved to:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
