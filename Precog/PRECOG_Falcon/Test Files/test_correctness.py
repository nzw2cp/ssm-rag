
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PRECOG import PRECOG


TOKEN_COUNTS = [2, 7, 13, 32]
GENERATE_TOKENS = 20

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


def difference(a, b):
    delta = (a.float() - b.float()).abs()
    return delta.max().item(), delta.mean().item()


def agreement(a, b):
    matches = a.argmax(-1) == b.argmax(-1)
    return matches.sum().item(), matches.numel()


@torch.inference_mode()
def generate(precog, state, logits, position):
    cache = precog.restore_snapshot(state)
    next_logits = logits[:, -1, :].to(precog.device)
    generated = []

    for step in range(GENERATE_TOKENS):
        token = next_logits.argmax(-1).item()
        generated.append(token)

        output = precog.forward(
            [token], cache=cache, position=position + step
        )
        next_logits = output.logits[:, -1, :]

    return generated, precog.snapshot(cache)


def test_generation(precog, ref, new, position):
    ref_tokens, ref_state = generate(
        precog, ref[1], ref[0], position
    )
    new_tokens, new_state = generate(
        precog, new[1], new[0], position
    )

    matches = sum(
        a == b for a, b in zip(ref_tokens, new_tokens)
    )
    conv, _ = difference(
        ref_state["conv_states"], new_state["conv_states"]
    )
    ssm, _ = difference(
        ref_state["ssm_states"], new_state["ssm_states"]
    )

    print("Sequential:", precog.tokenizer.decode(ref_tokens))
    print("Multi-token:", precog.tokenizer.decode(new_tokens))
    print(f"Generation match: {matches}/{GENERATE_TOKENS}")
    print(f"Final state difference: Conv {conv:.5f}, SSM {ssm:.5f}")

    return matches


def main():
    print("Falcon Mamba PRECOG correctness tests")

    precog = PRECOG()
    matched_total = token_total = generated_total = 0

    for name, document, question in SCENARIOS:
        context_length, snapshot = precog.prepare(document)
        question_ids = precog.tokens(question * 4)

        print(f"\n{name}: {context_length} context tokens")

        for count in TOKEN_COUNTS:
            ids = question_ids[:count]

            ref = precog.continue_tokens(
                snapshot, context_length, ids,
                method="sequential"
            )
            new = precog.continue_tokens(
                snapshot, context_length, ids,
                method="multitoken"
            )

            matched, total = agreement(ref[0], new[0])
            logits_max, logits_mean = difference(ref[0], new[0])
            conv, _ = difference(
                ref[1]["conv_states"], new[1]["conv_states"]
            )
            ssm, _ = difference(
                ref[1]["ssm_states"], new[1]["ssm_states"]
            )

            matched_total += matched
            token_total += total

            print(
                f"{count:2} tokens | Match {matched}/{total} | "
                f"Logit max {logits_max:.5f} | "
                f"Mean {logits_mean:.5f} | "
                f"Conv {conv:.5f} | SSM {ssm:.5f}"
            )

            if count == 32:
                generated_total += test_generation(
                    precog, ref, new, context_length + count
                )

    print("\nCorrectness summary")
    print(f"Predictions: {matched_total}/{token_total}")
    print(
        f"Generated tokens: "
        f"{generated_total}/{len(SCENARIOS) * GENERATE_TOKENS}"
    )
    print("Numerical equivalence is not guaranteed.")


if __name__ == "__main__":
    main()
