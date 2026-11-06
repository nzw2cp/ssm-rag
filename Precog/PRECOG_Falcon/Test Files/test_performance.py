
import csv
import statistics
from pathlib import Path

from test_correctness import (
    PRECOG,
    PRECOGMultiToken,
    SCENARIOS,
    TOKEN_COUNTS,
    prepare,
    run,
    agreement,
)


REPEATS = 3
RESULTS_FILE = Path(__file__).resolve().parent / "performance_results.csv"


def measure(continuation, snapshot, position, ids, method):
    # Warm up before collecting measurements.
    run(continuation, snapshot, position, ids, method)

    times = [
        run(continuation, snapshot, position, ids, method)[2]
        for _ in range(REPEATS)
    ]

    return statistics.median(times)


def main():
    print("Falcon Mamba PRECOG performance tests")

    precog = PRECOG()
    continuation = PRECOGMultiToken(precog)
    results = []

    for name, document, question in SCENARIOS:
        length, snapshot = prepare(precog, document)
        question_ids = precog.tokens(question * 4)

        print(f"\n{name}: {length} context tokens")

        for count in TOKEN_COUNTS:
            ids = question_ids[:count]

            reference, _, _ = run(
                continuation, snapshot, length, ids, "sequential"
            )
            candidate, _, _ = run(
                continuation, snapshot, length, ids, "multitoken"
            )

            matched, total = agreement(reference, candidate)

            sequential = measure(
                continuation, snapshot, length, ids, "sequential"
            )
            multitoken = measure(
                continuation, snapshot, length, ids, "multitoken"
            )

            speedup = sequential / multitoken

            results.append({
                "scenario": name,
                "context_tokens": length,
                "continuation_tokens": count,
                "matched": matched,
                "total": total,
                "sequential_seconds": round(sequential, 4),
                "multitoken_seconds": round(multitoken, 4),
                "speedup": round(speedup, 3),
            })

            print(
                f"{count:2} tokens | "
                f"Match {matched}/{total} | "
                f"Sequential {sequential:.4f}s | "
                f"Multi {multitoken:.4f}s | "
                f"Speedup {speedup:.2f}x",
                flush=True,
            )

    with RESULTS_FILE.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print("\nPerformance summary")
    print(
        "Median speedup:",
        f"{statistics.median(r['speedup'] for r in results):.2f}x",
    )
    print("Results saved to:", RESULTS_FILE)


if __name__ == "__main__":
    main()
