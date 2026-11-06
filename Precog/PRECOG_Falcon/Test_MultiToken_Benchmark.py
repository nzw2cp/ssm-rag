
import csv
import math
import statistics
import time
import traceback

import torch

import PRECOG_MultiToken as multi_module
from PRECOG import PRECOG
from PRECOG_MultiToken import PRECOGMultiToken


# Benchmark settings.
TOKEN_COUNTS = [2, 7, 13, 32]
TIMING_REPEATS = 3
WARMUP_RUNS = 1

OUTPUT_FILE = "multitoken_benchmark_results.csv"

# This is an analysis threshold, not proof of correctness.
STATE_TOLERANCE = 0.01


def quiet_debug_stage(layer, stage, operation):
    # Disable the diagnostic printing and synchronization
    # in PRECOG_MultiToken.py for performance testing.
    return operation()


multi_module.debug_stage = quiet_debug_stage


def synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def tensor_stats(reference, candidate):
    if reference.shape != candidate.shape:
        raise ValueError(
            f"Shape mismatch: {reference.shape} vs {candidate.shape}"
        )

    a = reference.float()
    b = candidate.float()

    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError("A tensor contains NaN or infinity")

    difference = (a - b).abs()

    return {
        "max": difference.max().item(),
        "mean": difference.mean().item(),
        "rmse": torch.sqrt(
            torch.mean((a - b) ** 2)
        ).item(),
    }


def prediction_stats(reference, candidate):
    ref_tokens = reference.argmax(dim=-1)
    new_tokens = candidate.argmax(dim=-1)

    matches = ref_tokens == new_tokens
    matched = int(matches.sum().item())
    total = matches.numel()

    # Calculate the gap between the top two reference logits.
    top_two = torch.topk(
        reference.float(),
        k=2,
        dim=-1,
    ).values

    margins = top_two[..., 0] - top_two[..., 1]

    mismatches = []

    for position in range(total):
        ref_id = int(ref_tokens.reshape(-1)[position])
        new_id = int(new_tokens.reshape(-1)[position])

        if ref_id != new_id:
            mismatches.append({
                "position": position + 1,
                "reference": ref_id,
                "multitoken": new_id,
                "reference_margin": float(
                    margins.reshape(-1)[position]
                ),
            })

    return {
        "matched": matched,
        "total": total,
        "agreement": matched / total,
        "minimum_margin": margins.min().item(),
        "mismatches": mismatches,
    }


def timed_run(
    continuation,
    snapshot,
    context_length,
    token_ids,
    method,
):
    synchronize()
    start = time.perf_counter()

    with torch.inference_mode():
        logits, final_state = continuation.continue_tokens(
            snapshot,
            context_length,
            token_ids,
            method=method,
        )

    synchronize()
    elapsed = time.perf_counter() - start

    return logits, final_state, elapsed


def benchmark_time(
    continuation,
    snapshot,
    context_length,
    token_ids,
    method,
):
    # Warm up relevant CUDA kernels before measuring.
    for _ in range(WARMUP_RUNS):
        timed_run(
            continuation,
            snapshot,
            context_length,
            token_ids,
            method,
        )

    synchronize()

    times = []

    for _ in range(TIMING_REPEATS):
        _, _, elapsed = timed_run(
            continuation,
            snapshot,
            context_length,
            token_ids,
            method,
        )

        times.append(elapsed)

    return {
        "median": statistics.median(times),
        "minimum": min(times),
        "maximum": max(times),
        "times": times,
    }


@torch.inference_mode()
def test_future_continuation(
    precog,
    continuation,
    reference_state,
    candidate_state,
    next_token,
    next_position,
):
    # Continue both final states with the same new token.
    reference_cache = continuation.restore_snapshot(
        reference_state
    )

    candidate_cache = continuation.restore_snapshot(
        candidate_state
    )

    reference_output = precog.forward(
        [next_token],
        cache=reference_cache,
        position=next_position,
        hidden_states=False,
    )

    candidate_output = precog.forward(
        [next_token],
        cache=candidate_cache,
        position=next_position,
        hidden_states=False,
    )

    reference_logits = reference_output.logits.float().cpu()
    candidate_logits = candidate_output.logits.float().cpu()

    difference = tensor_stats(
        reference_logits,
        candidate_logits,
    )

    predictions = prediction_stats(
        reference_logits,
        candidate_logits,
    )

    return difference, predictions


def print_tensor_stats(name, stats):
    print(
        f"{name}: "
        f"max={stats['max']:.8f}, "
        f"mean={stats['mean']:.8f}, "
        f"RMSE={stats['rmse']:.8f}"
    )


def prepare_context(precog, text):
    context_ids = precog.tokens(text)

    with torch.inference_mode():
        output = precog.forward(
            context_ids,
            hidden_states=False,
        )

        snapshot = precog.snapshot(
            output.cache_params
        )

    return context_ids, snapshot


def run_case(
    precog,
    continuation,
    scenario_name,
    context_ids,
    snapshot,
    question_ids,
    token_count,
):
    token_ids = question_ids[:token_count]
    context_length = len(context_ids)

    print(
        f"\nScenario: {scenario_name} | "
        f"Context: {context_length} tokens | "
        f"Continuation: {token_count} tokens",
        flush=True,
    )

    # First establish the outputs and states for both methods.
    reference_logits, reference_state, _ = timed_run(
        continuation,
        snapshot,
        context_length,
        token_ids,
        "sequential",
    )

    candidate_logits, candidate_state, _ = timed_run(
        continuation,
        snapshot,
        context_length,
        token_ids,
        "multitoken",
    )

    logits = tensor_stats(
        reference_logits,
        candidate_logits,
    )

    conv = tensor_stats(
        reference_state["conv_states"],
        candidate_state["conv_states"],
    )

    ssm = tensor_stats(
        reference_state["ssm_states"],
        candidate_state["ssm_states"],
    )

    predictions = prediction_stats(
        reference_logits,
        candidate_logits,
    )

    print(
        "Predictions matching:",
        f"{predictions['matched']}/{predictions['total']}",
    )

    print_tensor_stats("Logit difference", logits)
    print_tensor_stats("Convolution state difference", conv)
    print_tensor_stats("SSM state difference", ssm)

    print(
        "Minimum reference top-two logit margin:",
        f"{predictions['minimum_margin']:.8f}",
    )

    if predictions["mismatches"]:
        print("Prediction mismatches:")

        for mismatch in predictions["mismatches"]:
            print(
                f"  Position {mismatch['position']}: "
                f"{mismatch['reference']} vs "
                f"{mismatch['multitoken']}, "
                f"margin={mismatch['reference_margin']:.6f}"
            )

    # Verify that the new saved state can be reused.
    probe_ids = precog.tokens(" The answer is")
    next_token = probe_ids[0]

    future_diff, future_predictions = test_future_continuation(
        precog,
        continuation,
        reference_state,
        candidate_state,
        next_token,
        context_length + token_count,
    )

    print(
        "Future continuation prediction match:",
        future_predictions["matched"] == future_predictions["total"],
    )

    print_tensor_stats(
        "Future continuation logit difference",
        future_diff,
    )

    # Only benchmark after the correctness checks finish.
    print("Measuring execution times...", flush=True)

    sequential_time = benchmark_time(
        continuation,
        snapshot,
        context_length,
        token_ids,
        "sequential",
    )

    multitoken_time = benchmark_time(
        continuation,
        snapshot,
        context_length,
        token_ids,
        "multitoken",
    )

    baseline = sequential_time["median"]
    optimized = multitoken_time["median"]

    speedup = baseline / optimized if optimized > 0 else math.nan

    print(
        f"Sequential median: {baseline:.4f} seconds"
    )

    print(
        f"Multi-token median: {optimized:.4f} seconds"
    )

    print(
        f"Speedup: {speedup:.2f}x"
    )

    print(
        "Sequential runs:",
        [round(t, 4) for t in sequential_time["times"]],
    )

    print(
        "Multi-token runs:",
        [round(t, 4) for t in multitoken_time["times"]],
    )

    return {
        "scenario": scenario_name,
        "context_tokens": context_length,
        "continuation_tokens": token_count,
        "predictions_matched": predictions["matched"],
        "predictions_total": predictions["total"],
        "prediction_agreement": predictions["agreement"],
        "logit_max_difference": logits["max"],
        "logit_mean_difference": logits["mean"],
        "logit_rmse": logits["rmse"],
        "conv_max_difference": conv["max"],
        "ssm_max_difference": ssm["max"],
        "minimum_reference_margin": predictions["minimum_margin"],
        "future_prediction_match": (
            future_predictions["matched"]
            == future_predictions["total"]
        ),
        "future_logit_max_difference": future_diff["max"],
        "sequential_median_seconds": baseline,
        "multitoken_median_seconds": optimized,
        "speedup": speedup,
        "state_within_test_tolerance": (
            conv["max"] <= STATE_TOLERANCE
            and ssm["max"] <= STATE_TOLERANCE
        ),
    }


def save_results(rows):
    if not rows:
        return

    with open(
        OUTPUT_FILE,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(rows)

    print(f"\nResults saved to: {OUTPUT_FILE}")


def summarize(rows):
    print("\nBenchmark summary")

    print(
        f"{'Scenario':<16}"
        f"{'Context':>9}"
        f"{'Tokens':>9}"
        f"{'Agreement':>13}"
        f"{'Speedup':>12}"
    )

    total_matched = 0
    total_predictions = 0

    for row in rows:
        total_matched += row["predictions_matched"]
        total_predictions += row["predictions_total"]

        print(
            f"{row['scenario']:<16}"
            f"{row['context_tokens']:>9}"
            f"{row['continuation_tokens']:>9}"
            f"{row['prediction_agreement']:>12.1%}"
            f"{row['speedup']:>11.2f}x"
        )

    print(
        "\nTotal prediction agreement:",
        f"{total_matched}/{total_predictions}",
    )

    print(
        "Cases with reusable-state prediction agreement:",
        f"{sum(row['future_prediction_match'] for row in rows)}"
        f"/{len(rows)}",
    )

    print(
        "Cases within illustrative state tolerance:",
        f"{sum(row['state_within_test_tolerance'] for row in rows)}"
        f"/{len(rows)}",
    )

    print(
        "Median speedup across cases:",
        f"{statistics.median(row['speedup'] for row in rows):.2f}x",
    )


def main():
    print("Falcon Mamba multi-token correctness and speed benchmark")

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    else:
        print("CUDA unavailable")
        print("This experimental implementation requires CUDA")
        return

    print("PyTorch:", torch.__version__)

    precog = PRECOG()
    continuation = PRECOGMultiToken(precog)

    # Different document lengths, using the same factual topic.
    short_context = (
        "France is located in Europe. "
        "Its capital is Paris. "
        "Germany is located in Europe. "
        "Its capital is Berlin."
    )

    medium_context = (
        "France is a country in Western Europe. "
        "Paris is its capital and largest city. "
        "Germany has its capital in Berlin. "
        "Spain has its capital in Madrid. "
        "Italy has its capital in Rome. "
        "These countries belong to the European continent. "
    ) * 3

    long_context = (
        "France is a country in Western Europe. "
        "Its capital is Paris. "
        "Germany has its capital in Berlin. "
        "Spain has its capital in Madrid. "
        "Italy has its capital in Rome. "
        "Portugal has its capital in Lisbon. "
        "Austria has its capital in Vienna. "
        "Poland has its capital in Warsaw. "
        "The Netherlands has its capital in Amsterdam. "
        "Belgium has its capital in Brussels. "
    ) * 5

    scenarios = [
        ("Short", short_context),
        ("Medium", medium_context),
        ("Long", long_context),
    ]

    # Repeat the question to provide sufficient fixed input tokens.
    # These are teacher-forced input tokens, not generated answers.
    question = (
        " What is the capital of France? "
        "Please identify the capital using the document. "
        "What city is the capital of France? "
        "Answer using the provided information."
    )

    question_ids = precog.tokens(question)

    print("Available continuation tokens:", len(question_ids))

    results = []

    try:
        for scenario_name, context in scenarios:
            print(
                f"\nPreparing {scenario_name} document state...",
                flush=True,
            )

            context_ids, snapshot = prepare_context(
                precog,
                context,
            )

            print(
                "Document tokens:",
                len(context_ids),
            )

            for count in TOKEN_COUNTS:
                if count > len(question_ids):
                    print(
                        f"Skipping {count} tokens: "
                        "question is too short"
                    )
                    continue

                row = run_case(
                    precog,
                    continuation,
                    scenario_name,
                    context_ids,
                    snapshot,
                    question_ids,
                    count,
                )

                results.append(row)

    except Exception as error:
        print("\nBenchmark stopped due to an error")
        print("Error type:", type(error).__name__)
        print("Error:", str(error))
        traceback.print_exc()

        # Avoid more CUDA work after a potentially fatal GPU error.

    finally:
        save_results(results)

        if results:
            summarize(results)

    print("\nBenchmark finished")


if __name__ == "__main__":
    main()
