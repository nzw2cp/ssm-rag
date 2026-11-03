
import gc
import statistics
import time

import torch

from PRECOG import PRECOG


SENTENCE_COUNTS = [2, 4, 8]
REPEATS = 2
MIB = 1024 ** 2


def tensor_mib(tensor):
    return tensor.numel() * tensor.element_size() / MIB


def cache_mib(cache):
    return (
        tensor_mib(cache.conv_states)
        + tensor_mib(cache.ssm_states)
    )


def cleanup():
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


def measure_builtin(precog, all_tokens):
    cleanup()

    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()

    torch.cuda.synchronize()
    start = time.perf_counter()

    outputs = precog.forward(
        all_tokens,
        hidden_states=True,
    )

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    hidden_memory = sum(
        tensor_mib(tensor)
        for tensor in outputs.hidden_states
    )

    final_cache_memory = cache_mib(outputs.cache_params)

    peak_extra = max(
        0.0,
        (torch.cuda.max_memory_allocated() - baseline) / MIB,
    )

    result = {
        "time": elapsed,
        "peak_gpu": peak_extra,
        "hidden_memory": hidden_memory,
        "final_cache": final_cache_memory,
    }

    del outputs
    cleanup()

    return result


def measure_precog(precog, sentence_tokens):
    cleanup()

    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()

    cache = None
    position = 0
    snapshots = []

    torch.cuda.synchronize()
    start = time.perf_counter()

    for tokens in sentence_tokens:
        if cache is None:
            outputs = precog.forward(
                tokens,
                hidden_states=False,
            )

            cache = outputs.cache_params
            position = len(tokens)
            del outputs

        else:
            # Continue from the previous sentence one token at a time.
            for token_id in tokens:
                outputs = precog.forward(
                    [token_id],
                    cache=cache,
                    position=position,
                    hidden_states=False,
                )

                position += 1
                del outputs

        # Keep a CPU copy of the state after each sentence.
        snapshots.append({
            "conv": cache.conv_states.detach().cpu().clone(),
            "ssm": cache.ssm_states.detach().cpu().clone(),
        })

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    all_snapshots_memory = sum(
        tensor_mib(item["conv"])
        + tensor_mib(item["ssm"])
        for item in snapshots
    )

    final_cache_memory = cache_mib(cache)

    peak_extra = max(
        0.0,
        (torch.cuda.max_memory_allocated() - baseline) / MIB,
    )

    result = {
        "time": elapsed,
        "peak_gpu": peak_extra,
        "all_snapshots": all_snapshots_memory,
        "final_cache": final_cache_memory,
    }

    del cache, snapshots
    cleanup()

    return result


def repeat_measurement(function, *args):
    results = [
        function(*args)
        for _ in range(REPEATS)
    ]

    return {
        key: statistics.median(
            result[key] for result in results
        )
        for key in results[0]
    }


def print_results(count, token_count, builtin, precog):
    print(f"\nResults for {count} sentences and {token_count} tokens\n")

    print(
        f"{'Measurement':<34}"
        f"{'Built-in':>15}"
        f"{'PRECOG':>15}"
    )

    print(
        f"{'Execution time (seconds)':<34}"
        f"{builtin['time']:>15.4f}"
        f"{precog['time']:>15.4f}"
    )

    print(
        f"{'Peak extra GPU memory (MiB)':<34}"
        f"{builtin['peak_gpu']:>15.2f}"
        f"{precog['peak_gpu']:>15.2f}"
    )

    print(
        f"{'Full hidden activations (MiB)':<34}"
        f"{builtin['hidden_memory']:>15.2f}"
        f"{'Not retained':>15}"
    )

    print(
        f"{'Final recurrent state (MiB)':<34}"
        f"{builtin['final_cache']:>15.2f}"
        f"{precog['final_cache']:>15.2f}"
    )

    print(
        f"{'All sentence snapshots (MiB)':<34}"
        f"{'Not retained':>15}"
        f"{precog['all_snapshots']:>15.2f}"
    )

    print("\nComparison")

    ratio = builtin["time"] / precog["time"]

    if ratio > 1:
        print(f"PRECOG was {ratio:.2f}x faster")
    else:
        print(f"Built-in extraction was {1 / ratio:.2f}x faster")

    print(
        "Hidden activations compared to final PRECOG state: "
        f"{builtin['hidden_memory'] / precog['final_cache']:.2f}x"
    )

    print(
        "Hidden activations compared to all sentence snapshots: "
        f"{builtin['hidden_memory'] / precog['all_snapshots']:.2f}x"
    )


def main():
    print("Falcon Mamba hidden-state comparison")
    print("Built-in Hidden States vs PRECOG")

    precog = PRECOG()

    sentences = [
        (
            f"Sentence {i} describes how organizations "
            f"process information and improve operational "
            f"efficiency through computer systems."
        )
        for i in range(1, max(SENTENCE_COUNTS) + 1)
    ]

    prepared = []

    for i, sentence in enumerate(sentences):
        text = sentence if i == 0 else " " + sentence
        prepared.append(precog.tokens(text))

    print("\nWarming up GPU...")

    warmup_ids = precog.tokens(
        "This is a short warmup sentence."
    )

    result = precog.forward(
        warmup_ids,
        hidden_states=True,
    )
    del result

    result = precog.forward(warmup_ids)
    cache = result.cache_params
    del result

    result = precog.forward(
        [warmup_ids[0]],
        cache=cache,
        position=len(warmup_ids),
        hidden_states=False,
    )

    del result, cache
    cleanup()

    print("\nStarting comparison...")

    for sentence_count in SENTENCE_COUNTS:
        sentence_tokens = prepared[:sentence_count]

        all_tokens = [
            token
            for sentence in sentence_tokens
            for token in sentence
        ]

        print(
            f"\nMeasuring {sentence_count} sentences "
            f"with {len(all_tokens)} tokens..."
        )

        builtin = repeat_measurement(
            measure_builtin,
            precog,
            all_tokens,
        )

        precog_result = repeat_measurement(
            measure_precog,
            precog,
            sentence_tokens,
        )

        print_results(
            sentence_count,
            len(all_tokens),
            builtin,
            precog_result,
        )

    print("\nComparison complete")
    print(f"Results use the median of {REPEATS} runs")


if __name__ == "__main__":
    main()
