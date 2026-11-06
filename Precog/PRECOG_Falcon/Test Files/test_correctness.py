
import sys
import time
from pathlib import Path

import torch

# Allow Current File execution from the Test Files folder.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import PRECOG_MultiToken as multi
from PRECOG import PRECOG
from PRECOG_MultiToken import PRECOGMultiToken


# Disable the old per-layer debugging output.
multi.debug_stage = lambda layer, stage, operation: operation()

TOKEN_COUNTS = [2, 7, 13, 32]

SCENARIOS = [
    (
        "Geography",
        "France has its capital in Paris. Germany has its capital "
        "in Berlin. Italy has its capital in Rome. " * 3,
        " What is the capital of France? Explain using the "
        "provided information. ",
    ),
    (
        "Computing",
        "A CPU executes instructions. RAM stores temporary data. "
        "A scheduler determines which process uses the CPU. "
        "A cache stores frequently accessed information. " * 3,
        " What does a CPU scheduler do? Explain the role of "
        "scheduling in an operating system. ",
    ),
]


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def prepare(precog, text):
    ids = precog.tokens(text)

    with torch.inference_mode():
        output = precog.forward(ids, hidden_states=False)
        snapshot = precog.snapshot(output.cache_params)

    return len(ids), snapshot


def run(continuation, snapshot, position, ids, method):
    sync()
    start = time.perf_counter()

    with torch.inference_mode():
        logits, state = continuation.continue_tokens(
            snapshot, position, ids, method=method
        )

    sync()
    return logits, state, time.perf_counter() - start


def difference(a, b):
    delta = (a.float() - b.float()).abs()
    return delta.max().item(), delta.mean().item()


def agreement(a, b):
    matches = a.argmax(-1) == b.argmax(-1)
    return int(matches.sum()), matches.numel()


def future_check(precog, continuation, states, position, ids):
    outputs = []

    with torch.inference_mode():
        for state in states:
            cache = continuation.restore_snapshot(state)
            logits = []

            for offset, token in enumerate(ids):
                result = precog.forward(
                    [token],
                    cache=cache,
                    position=position + offset,
                    hidden_states=False,
                )
                logits.append(result.logits.float().cpu())

            outputs.append(torch.cat(logits, dim=1))

    return agreement(*outputs)


def main():
    print("Falcon Mamba PRECOG correctness tests")

    precog = PRECOG()
    continuation = PRECOGMultiToken(precog)

    total_matches = 0
    total_tokens = 0
    future_matches = 0
    future_tokens = 0

    for name, document, question in SCENARIOS:
        length, snapshot = prepare(precog, document)
        question_ids = precog.tokens(question * 4)

        print(f"\n{name}: {length} context tokens")

        for count in TOKEN_COUNTS:
            ids = question_ids[:count]

            ref, ref_state, _ = run(
                continuation, snapshot, length, ids, "sequential"
            )
            new, new_state, _ = run(
                continuation, snapshot, length, ids, "multitoken"
            )

            matched, total = agreement(ref, new)
            logit_max, logit_mean = difference(ref, new)

            conv_max, _ = difference(
                ref_state["conv_states"], new_state["conv_states"]
            )
            ssm_max, _ = difference(
                ref_state["ssm_states"], new_state["ssm_states"]
            )

            # Continue both saved states with identical extra tokens.
            extra = question_ids[count:count + 3]
            next_matched, next_total = future_check(
                precog,
                continuation,
                (ref_state, new_state),
                length + count,
                extra,
            )

            total_matches += matched
            total_tokens += total
            future_matches += next_matched
            future_tokens += next_total

            print(
                f"{count:2} tokens | "
                f"Match {matched}/{total} | "
                f"Future {next_matched}/{next_total} | "
                f"Logit max {logit_max:.5f} | "
                f"Logit mean {logit_mean:.5f} | "
                f"Conv {conv_max:.5f} | "
                f"SSM {ssm_max:.5f}"
            )

    print("\nCorrectness summary")
    print(f"Predictions: {total_matches}/{total_tokens}")
    print(f"Future predictions: {future_matches}/{future_tokens}")
    print("Note: prediction agreement does not prove exact equivalence.")


if __name__ == "__main__":
    main()
