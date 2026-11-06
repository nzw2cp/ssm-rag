
import gc
import math
import statistics
import time

import torch

from PRECOG import PRECOG


SENTENCE_COUNTS = [2, 4, 8]
REPEATS = 3
MIB = 1024 ** 2

QUESTIONS = [
    "What do the sentences describe?",
    "How can organizations improve operational efficiency?",
    "What role do computer systems play?",
]


def tensor_mib(tensor):
    return tensor.numel() * tensor.element_size() / MIB


def cache_mib(cache):
    return (
        tensor_mib(cache.conv_states)
        + tensor_mib(cache.ssm_states)
    )


def snapshot_mib(snapshot):
    return (
        tensor_mib(snapshot["conv_states"])
        + tensor_mib(snapshot["ssm_states"])
    )


def synchronize():
    torch.cuda.synchronize()


def cleanup():
    gc.collect()
    torch.cuda.empty_cache()


def start_measurement():
    cleanup()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    synchronize()
    return baseline, time.perf_counter()


def peak_memory(baseline):
    return max(
        0.0,
        (torch.cuda.max_memory_allocated() - baseline) / MIB,
    )


def median_result(results, key):
    return statistics.median(
        result[key] for result in results
    )


def make_sentences(count):
    return [
        (
            f"Sentence {i} describes how organizations "
            "process information and improve operational "
            "efficiency through computer systems."
        )
        for i in range(1, count + 1)
    ]


def make_context(precog, count):
    sentences = make_sentences(count)

    text = "Context:\n" + " ".join(sentences) + "\n"

    return precog.tokens(text)


def make_question(precog, question):
    return precog.tokens(
        f"\nQuestion: {question}\nAnswer:"
    )


def measure_builtin_extraction(precog, context_ids):
    baseline, start = start_measurement()

    outputs = precog.forward(
        context_ids,
        hidden_states=True,
    )

    synchronize()
    elapsed = time.perf_counter() - start

    hidden_memory = sum(
        tensor_mib(tensor)
        for tensor in outputs.hidden_states
    )

    final_cache_memory = cache_mib(outputs.cache_params)
    gpu_peak = peak_memory(baseline)

    result = {
        "time": elapsed,
        "peak_gpu": gpu_peak,
        "hidden_memory": hidden_memory,
        "final_cache": final_cache_memory,
    }

    del outputs
    cleanup()

    return result


def measure_precog_preparation(precog, context_ids):
    baseline, start = start_measurement()

    # Process the full document in one forward pass.
    outputs = precog.forward(
        context_ids,
        hidden_states=False,
    )

    synchronize()
    forward_time = time.perf_counter() - start

    # Store one final recurrent state on the CPU.
    copy_start = time.perf_counter()

    snapshot = precog.snapshot(outputs.cache_params)

    synchronize()
    copy_time = time.perf_counter() - copy_start

    total_time = time.perf_counter() - start

    gpu_peak = peak_memory(baseline)

    result = {
        "time": total_time,
        "forward_time": forward_time,
        "copy_time": copy_time,
        "peak_gpu": gpu_peak,
        "snapshot_memory": snapshot_mib(snapshot),
        "final_cache": cache_mib(outputs.cache_params),
    }

    del outputs
    cleanup()

    return result, snapshot


def restore_snapshot(precog, snapshot):
    cache = precog.new_cache()

    with torch.no_grad():
        cache.conv_states.copy_(
            snapshot["conv_states"].to(
                device=cache.conv_states.device,
                dtype=cache.conv_states.dtype,
            )
        )

        cache.ssm_states.copy_(
            snapshot["ssm_states"].to(
                device=cache.ssm_states.device,
                dtype=cache.ssm_states.dtype,
            )
        )

    return cache


def measure_builtin_query(precog, context_ids, question_ids):
    # Baseline processes the entire context for every query.
    full_ids = context_ids + question_ids

    baseline, start = start_measurement()

    outputs = precog.forward(
        full_ids,
        hidden_states=False,
    )

    synchronize()

    logits = outputs.logits[:, -1, :].float().cpu().clone()

    synchronize()
    elapsed = time.perf_counter() - start

    gpu_peak = peak_memory(baseline)

    result = {
        "time": elapsed,
        "peak_gpu": gpu_peak,
        "logits": logits,
    }

    del outputs
    cleanup()

    return result


def measure_precog_query(
    precog,
    snapshot,
    context_length,
    question_ids,
):
    baseline, start = start_measurement()

    # Restore the previously computed document state.
    cache = restore_snapshot(precog, snapshot)

    synchronize()
    restore_time = time.perf_counter() - start

    # Continue from the saved document state.
    token_start = time.perf_counter()
    position = context_length
    outputs = None

    for token_id in question_ids:
        outputs = precog.forward(
            [token_id],
            cache=cache,
            position=position,
            hidden_states=False,
        )

        position += 1

    synchronize()

    logits = outputs.logits[:, -1, :].float().cpu().clone()

    synchronize()
    continuation_time = time.perf_counter() - token_start
    total_time = time.perf_counter() - start

    gpu_peak = peak_memory(baseline)

    result = {
        "time": total_time,
        "restore_time": restore_time,
        "continuation_time": continuation_time,
        "peak_gpu": gpu_peak,
        "logits": logits,
    }

    del outputs, cache
    cleanup()

    return result


def compare_outputs(builtin_logits, precog_logits):
    difference = (
        builtin_logits - precog_logits
    ).abs()

    builtin_token = int(
        builtin_logits.argmax(dim=-1).item()
    )

    precog_token = int(
        precog_logits.argmax(dim=-1).item()
    )

    return {
        "max_difference": difference.max().item(),
        "mean_difference": difference.mean().item(),
        "tokens_match": builtin_token == precog_token,
        "builtin_token": builtin_token,
        "precog_token": precog_token,
    }


def test_preparation(precog, context_ids):
    builtin_results = []
    precog_results = []
    snapshot = None

    for _ in range(REPEATS):
        builtin_results.append(
            measure_builtin_extraction(
                precog,
                context_ids,
            )
        )

        preparation, snapshot = measure_precog_preparation(
            precog,
            context_ids,
        )

        precog_results.append(preparation)

    builtin = {
        key: median_result(builtin_results, key)
        for key in builtin_results[0]
    }

    prepared = {
        key: median_result(precog_results, key)
        for key in precog_results[0]
    }

    return builtin, prepared, snapshot


def test_repeated_queries(precog, context_ids, snapshot):
    results = []

    for question in QUESTIONS:
        question_ids = make_question(precog, question)

        builtin_runs = []
        precog_runs = []

        for _ in range(REPEATS):
            builtin_runs.append(
                measure_builtin_query(
                    precog,
                    context_ids,
                    question_ids,
                )
            )

            precog_runs.append(
                measure_precog_query(
                    precog,
                    snapshot,
                    len(context_ids),
                    question_ids,
                )
            )

        builtin_time = median_result(
            builtin_runs,
            "time",
        )

        precog_time = median_result(
            precog_runs,
            "time",
        )

        restore_time = median_result(
            precog_runs,
            "restore_time",
        )

        continuation_time = median_result(
            precog_runs,
            "continuation_time",
        )

        builtin_gpu = median_result(
            builtin_runs,
            "peak_gpu",
        )

        precog_gpu = median_result(
            precog_runs,
            "peak_gpu",
        )

        # The same model and input tokens are used for both.
        equivalence = compare_outputs(
            builtin_runs[-1]["logits"],
            precog_runs[-1]["logits"],
        )

        results.append({
            "question": question,
            "question_tokens": len(question_ids),
            "builtin_time": builtin_time,
            "precog_time": precog_time,
            "restore_time": restore_time,
            "continuation_time": continuation_time,
            "builtin_gpu": builtin_gpu,
            "precog_gpu": precog_gpu,
            **equivalence,
        })

    return results


def print_preparation(builtin, prepared):
    print("\nInitial state preparation")
    print(
        f"{'Measurement':<34}"
        f"{'Built-in':>15}"
        f"{'PRECOG':>15}"
    )

    print(
        f"{'Total time (seconds)':<34}"
        f"{builtin['time']:>15.4f}"
        f"{prepared['time']:>15.4f}"
    )

    print(
        f"{'Peak extra GPU memory (MiB)':<34}"
        f"{builtin['peak_gpu']:>15.2f}"
        f"{prepared['peak_gpu']:>15.2f}"
    )

    print(
        f"{'Hidden activations (MiB)':<34}"
        f"{builtin['hidden_memory']:>15.2f}"
        f"{'Not retained':>15}"
    )

    print(
        f"{'Saved recurrent state (MiB)':<34}"
        f"{'Not retained':>15}"
        f"{prepared['snapshot_memory']:>15.2f}"
    )

    print(
        f"\nPRECOG document forward time: "
        f"{prepared['forward_time']:.4f} seconds"
    )

    print(
        f"PRECOG CPU state-copy time: "
        f"{prepared['copy_time']:.4f} seconds"
    )


def print_query_results(results):
    print("\nRepeated query results")

    for number, result in enumerate(results, 1):
        print(f"\nQuestion {number}: {result['question']}")
        print(f"Question tokens: {result['question_tokens']}")

        print(
            f"Built-in time: "
            f"{result['builtin_time']:.4f} seconds"
        )

        print(
            f"PRECOG time: "
            f"{result['precog_time']:.4f} seconds"
        )

        print(
            f"PRECOG restore time: "
            f"{result['restore_time']:.4f} seconds"
        )

        print(
            f"PRECOG continuation time: "
            f"{result['continuation_time']:.4f} seconds"
        )

        print(
            f"Built-in peak GPU memory: "
            f"{result['builtin_gpu']:.2f} MiB"
        )

        print(
            f"PRECOG peak GPU memory: "
            f"{result['precog_gpu']:.2f} MiB"
        )

        print(
            f"Maximum logit difference: "
            f"{result['max_difference']:.6f}"
        )

        print(
            f"Mean logit difference: "
            f"{result['mean_difference']:.6f}"
        )

        print(
            f"Predicted tokens match: "
            f"{result['tokens_match']}"
        )


def print_summary(results, preparation_time):
    builtin_average = statistics.mean(
        result["builtin_time"]
        for result in results
    )

    precog_average = statistics.mean(
        result["precog_time"]
        for result in results
    )

    matches = sum(
        result["tokens_match"]
        for result in results
    )

    print("\nOverall comparison")

    print(
        f"Average built-in query time: "
        f"{builtin_average:.4f} seconds"
    )

    print(
        f"Average PRECOG query time: "
        f"{precog_average:.4f} seconds"
    )

    print(
        f"Matching predictions: "
        f"{matches}/{len(results)}"
    )

    if precog_average < builtin_average:
        speedup = builtin_average / precog_average

        print(
            f"PRECOG query speedup: {speedup:.2f}x"
        )

        savings = builtin_average - precog_average

        break_even = math.ceil(
            preparation_time / savings
        )

        print(
            f"Estimated break-even queries: {break_even}"
        )

    else:
        speedup = precog_average / builtin_average

        print(
            f"Built-in queries are {speedup:.2f}x faster"
        )

        print(
            "PRECOG does not break even at the "
            "measured query speeds"
        )


def warmup(precog):
    print("\nWarming up GPU...")

    ids = precog.tokens(
        "This is a short warmup sentence."
    )

    output = precog.forward(
        ids,
        hidden_states=True,
    )

    cache = output.cache_params

    precog.forward(
        [ids[0]],
        cache=cache,
        position=len(ids),
        hidden_states=False,
    )

    del output, cache
    cleanup()


def main():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "This comparison requires a CUDA GPU"
        )

    print("Falcon Mamba PRECOG comparison")
    print("Built-in Hidden States vs PRECOG")
    print(f"Repeats per measurement: {REPEATS}")

    precog = PRECOG()
    warmup(precog)

    for count in SENTENCE_COUNTS:
        context_ids = make_context(precog, count)

        print(
            f"\nResults for {count} sentences "
            f"and {len(context_ids)} context tokens"
        )

        builtin, prepared, snapshot = test_preparation(
            precog,
            context_ids,
        )

        print_preparation(builtin, prepared)

        query_results = test_repeated_queries(
            precog,
            context_ids,
            snapshot,
        )

        print_query_results(query_results)

        print_summary(
            query_results,
            prepared["time"],
        )

        del snapshot
        cleanup()

    print("\nComparison complete")


if __name__ == "__main__":
    main()
