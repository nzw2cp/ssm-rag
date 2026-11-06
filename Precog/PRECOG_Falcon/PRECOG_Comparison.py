
# Research benchmark: full-context processing versus restored PRECOG states.

import csv
import gc
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from PRECOG import PRECOG

RESULTS = ROOT / "Test Files" / "results"
REPEATS = 3
MULTIPLIERS = (1, 4, 8)

DOCUMENTS = (
    (
        "Geography",
        "France has its capital in Paris. "
        "Germany has its capital in Berlin. "
        "Italy has its capital in Rome. ",
        " What is the capital of France?",
    ),
    (
        "Computing",
        "A CPU executes instructions. RAM stores temporary data. "
        "A scheduler determines which process uses the CPU. ",
        " What does a CPU scheduler do?",
    ),
)


def measure(fn):
    # Track extra peak allocated GPU memory above existing allocations.
    gc.collect()
    torch.cuda.synchronize()
    start_memory = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()

    output = fn()

    torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    peak = (
        torch.cuda.max_memory_allocated() - start_memory
    ) / 1024**2

    return output, seconds, max(peak, 0.0)


def repeat(fn):
    # Warm up before measuring.
    measure(fn)

    samples = [
        measure(fn) for _ in range(REPEATS)
    ]

    return (
        statistics.median(x[1] for x in samples),
        max(x[2] for x in samples),
        [x[1] for x in samples],
    )


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    model = PRECOG()
    rows, runs = [], []

    try:
        for topic, document, question in DOCUMENTS:
            question_ids = model.tokens(question)

            for scale in MULTIPLIERS:
                context_ids = model.tokens(document * scale)
                length = len(context_ids)

                # PRECOG's document encoding is an offline cost.
                (_, state), encode_s, encode_mib = measure(
                    lambda: model.prepare(context_ids)
                )

                state_mib = sum(
                    state[k].numel() * state[k].element_size()
                    for k in ("conv_states", "ssm_states")
                ) / 1024**2

                # Baseline processes document and question together.
                def baseline():
                    output = model.forward(context_ids + question_ids)
                    return output.logits[
                        :, -len(question_ids):
                    ].float().cpu().clone()

                # PRECOG restores previously computed recurrent states.
                def precog():
                    return model.continue_tokens(
                        state, length, question_ids, "multitoken"
                    )[0]

                left, right = baseline(), precog()

                agreement = int(
                    (left.argmax(-1) == right.argmax(-1)).sum()
                )
                total = left.shape[1]
                logit_max = float((left - right).abs().max())

                base_s, base_mib, b_runs = repeat(baseline)
                cached_s, cached_mib, c_runs = repeat(precog)

                row = {
                    "topic": topic,
                    "context_tokens": length,
                    "question_tokens": total,
                    "baseline_s": base_s,
                    "precog_s": cached_s,
                    "speedup": base_s / cached_s,
                    "baseline_peak_mib": base_mib,
                    "precog_peak_mib": cached_mib,
                    "encoding_s": encode_s,
                    "encoding_peak_mib": encode_mib,
                    "stored_state_mib": state_mib,
                    "matched": agreement,
                    "total": total,
                    "logit_max": logit_max,
                }

                rows.append(row)

                runs.append({
                    "topic": topic,
                    "context_tokens": length,
                    "baseline_runs": b_runs,
                    "precog_runs": c_runs,
                })

                print(
                    f"{topic} | context {length} | "
                    f"{base_s:.3f}s vs {cached_s:.3f}s | "
                    f'{row["speedup"]:.2f}x | '
                    f"GPU +{base_mib:.1f}/{cached_mib:.1f} MiB | "
                    f"predictions {agreement}/{total}",
                    flush=True,
                )

    finally:
        # Save completed results even if a later test fails.
        if rows:
            path = RESULTS / "comparison.csv"

            with path.open(
                "w", newline="", encoding="utf-8"
            ) as file:
                writer = csv.DictWriter(
                    file, fieldnames=rows[0].keys()
                )
                writer.writeheader()
                writer.writerows(rows)

            print("Saved:", path)

        (RESULTS / "comparison_runs.json").write_text(
            json.dumps({
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "model": model.model_name,
                "gpu": torch.cuda.get_device_name(0),
                "repeats": REPEATS,
                "cases": runs,
            }, indent=2),
            encoding="utf-8",
        )

    if rows:
        median = statistics.median(
            r["speedup"] for r in rows
        )

        print(f"Median online PRECOG speedup: {median:.2f}x")
        print(
            "Online results exclude document encoding. "
            "Offline cost and state size are reported separately."
        )
        print(
            "GPU memory values are additional peak PyTorch "
            "allocations, not total VRAM usage."
        )


if __name__ == "__main__":
    main()
