
# Test independent state souping and compare against document concatenation.

import csv
import statistics
import time
from pathlib import Path

import torch

from Souping import Souping


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

DOCUMENTS = [
    "In the fictional city of Bellhaven, engineer Mira Solen created Project Lantern.",
    "Project Lantern was developed to clean drinking water for the village of Ardon.",
    "The village of Ardon is located beside the eastern Kestrel River.",
]

QUESTION = (
    "\nQuestion: What project did Mira Solen create, "
    "and what was its purpose?\nAnswer:"
)

METHODS = ("average", "sum", "max")
GENERATE_TOKENS = 40


def timed(fn):
    torch.cuda.synchronize()
    start = time.perf_counter()

    result = fn()

    torch.cuda.synchronize()
    return result, time.perf_counter() - start


def test_merging(soup, states):
    # One-document pooling must preserve every state exactly.
    single = soup.merge(states[:1], "average")

    for name in ("conv_states", "ssm_states"):
        assert torch.equal(single[name], states[0][name])

    assert single["seqlen_offset"] == states[0]["seqlen_offset"]
    print("Single-document state identity: PASS")

    # Pooling should not meaningfully depend on document order.
    forward = soup.merge(states, "average")
    reversed_order = soup.merge(states[::-1], "average")

    for name in ("conv_states", "ssm_states"):
        assert torch.allclose(
            forward[name].float(),
            reversed_order[name].float(),
            atol=1e-2,
            rtol=1e-3,
        )

    print("Average pooling order check: PASS")

    # Check that all pooling methods produce finite states.
    for method in METHODS:
        state = soup.merge(states, method)

        for name in ("conv_states", "ssm_states"):
            assert torch.isfinite(state[name]).all()

    print("Pooling validity: PASS")


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)

    soup = Souping()

    print("\nEncoding documents independently...")

    states = []
    encoding_times = []

    for index, document in enumerate(DOCUMENTS, 1):
        state, elapsed = timed(lambda: soup.encode(document))

        states.append(state)
        encoding_times.append(elapsed)

        print(
            f"Document {index}: "
            f"{state['seqlen_offset']} tokens | "
            f"{elapsed:.4f}s"
        )

    test_merging(soup, states)

    rows = []

    # Use concatenation as the reference approach.
    variants = ("concat",) + METHODS

    for method in variants:
        print(f"\nTesting {method.upper()}...")

        if method == "concat":
            combined = "\n".join(DOCUMENTS)
            state, preparation = timed(lambda: soup.encode(combined))
            encoding_cost = preparation
        else:
            state, preparation = timed(
                lambda: soup.merge(states, method)
            )
            encoding_cost = sum(encoding_times)

        answer, online_time = timed(
            lambda: soup.answer(
                state, QUESTION, max_new_tokens=GENERATE_TOKENS
            )
        )

        # These checks are only a simple diagnostic, not an
        # established question-answering evaluation metric.
        lower_answer = answer.lower()
        contains_project = "lantern" in lower_answer
        contains_purpose = "water" in lower_answer

        row = {
            "method": method,
            "document_count": len(DOCUMENTS),
            "state_offset": state["seqlen_offset"],
            "state_size_mib": round(soup.state_size_mib(state), 4),
            "encoding_seconds": round(encoding_cost, 6),
            "merge_seconds": round(preparation, 6)
                if method != "concat" else 0,
            "online_seconds": round(online_time, 6),
            "contains_project": contains_project,
            "contains_purpose": contains_purpose,
            "answer": answer,
        }

        rows.append(row)

        print("Answer:", answer or "(empty)")
        print(f"Online time: {online_time:.4f}s")
        print(f"State size: {row['state_size_mib']:.4f} MiB")
        print(
            "Contains expected information:",
            f"Project={contains_project}, Purpose={contains_purpose}"
        )

    path = RESULTS / "souping_results.csv"

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file, fieldnames=list(rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\nSouping comparison")
    print(f"{'Method':<10} {'Online (s)':>12} {'Project':>10} {'Purpose':>10}")

    for row in rows:
        print(
            f"{row['method']:<10} "
            f"{row['online_seconds']:>12.4f} "
            f"{str(row['contains_project']):>10} "
            f"{str(row['contains_purpose']):>10}"
        )

    print("\nSaved:", path)
    print("Independent encoding is a one-time cost.")
    print("These are exploratory tests, not final QA benchmarks.")


if __name__ == "__main__":
    main()
