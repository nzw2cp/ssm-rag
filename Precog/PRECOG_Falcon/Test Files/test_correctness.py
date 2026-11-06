
# PRECOG regression checks: multi-token outputs, generation, and saved-state features.

import json
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PRECOG import PRECOG

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
COUNTS = (2, 7, 13, 32)

SCENARIOS = (
    (
        "Geography",
        "France has its capital in Paris. Germany has its capital in Berlin. "
        "Italy has its capital in Rome. " * 3,
        " What is the capital of France? Explain using the provided information. ",
    ),
    (
        "Computing",
        "A CPU executes instructions. RAM stores temporary data. "
        "A scheduler determines which process uses the CPU. "
        "A cache stores frequently accessed information. " * 3,
        " What does a CPU scheduler do? Explain the role of scheduling in an operating system. ",
    ),
)


def match(a, b):
    equal = a.argmax(-1) == b.argmax(-1)
    return int(equal.sum()), equal.numel()


def maxdiff(a, b):
    return float((a.float() - b.float()).abs().max())


@torch.inference_mode()
def generate(model, output, position, count=20):
    logits, state = output
    cache = model.restore_snapshot(state)
    next_logits = logits[:, -1, :].to(model.device)
    tokens = []

    for i in range(count):
        token = int(next_logits.argmax(-1).item())
        tokens.append(token)
        next_logits = model.forward(
            [token], cache=cache, position=position + i
        ).logits[:, -1, :]

    return tokens


def smoke(model):
    # Use temporary directories without modifying existing indexed documents.
    with tempfile.TemporaryDirectory() as temp:
        old = (model.storage_dir, model.manifest_path, model.manifest)

        try:
            model.storage_dir = Path(temp) / "states"
            model.storage_dir.mkdir()
            model.manifest_path = model.storage_dir / "manifest.json"
            model.manifest = {}

            text = "The capital of France is Paris."
            model.index_documents({
                "france": text,
                "germany": "Berlin is in Germany.",
            })

            assert set(model.list_chunks()) == {"france", "germany"}

            restored = model.restore_chunk("france")
            length, state = model.prepare(text)

            assert restored.seqlen_offset == length

            for key in ("conv_states", "ssm_states"):
                assert torch.equal(
                    getattr(restored, key).cpu(), state[key]
                )

            question = " What is its capital?"
            direct = model.continue_tokens(state, length, question)[0]
            indexed = model.continue_from_chunk("france", question)

            assert torch.allclose(
                direct, indexed, atol=1e-3, rtol=1e-3
            )

            exported = model.export_hidden_states(
                "Paris is in France.",
                output_dir=Path(temp) / "exports",
            )

            assert (exported / "final_states.txt").stat().st_size > 0
            assert (exported / "layer_activations.txt").stat().st_size > 0
            assert len(list(
                (exported / "intermediate_states").glob("token_*.txt")
            )) == len(model.tokens("Paris is in France."))

            hidden = model.extract_hidden_states("Paris is in France.")
            assert len(hidden["hidden_states"]) == (
                model.model.config.num_hidden_layers + 1
            )

            return True

        finally:
            model.storage_dir, model.manifest_path, model.manifest = old


def main():
    RESULTS.mkdir(exist_ok=True)
    model = PRECOG()
    rows, generations = [], []

    report = {
        "cases": rows,
        "generation": generations,
        "smoke_passed": False,
    }

    try:
        for topic, document, question in SCENARIOS:
            length, state = model.prepare(document)
            tokens = model.tokens(question * 4)

            for count in COUNTS:
                a = model.continue_tokens(
                    state, length, tokens[:count], "sequential"
                )
                b = model.continue_tokens(
                    state, length, tokens[:count], "multitoken"
                )

                matched, total = match(a[0], b[0])

                row = {
                    "topic": topic,
                    "context_tokens": length,
                    "tokens": count,
                    "matched": matched,
                    "total": total,
                    "logit_max": maxdiff(a[0], b[0]),
                    "conv_max": maxdiff(
                        a[1]["conv_states"], b[1]["conv_states"]
                    ),
                    "ssm_max": maxdiff(
                        a[1]["ssm_states"], b[1]["ssm_states"]
                    ),
                }

                rows.append(row)
                print(
                    f'{topic}: {count} tokens | {matched}/{total} | '
                    f'logit max {row["logit_max"]:.5f}'
                )

                if count == 32:
                    left = generate(model, a, length + count)
                    right = generate(model, b, length + count)
                    same = sum(x == y for x, y in zip(left, right))

                    generations.append({
                        "topic": topic,
                        "matched": same,
                        "total": 20,
                        "sequential_text": model.tokenizer.decode(left),
                        "multitoken_text": model.tokenizer.decode(right),
                    })

                    print(f"  Generated: {same}/20")

        report["smoke_passed"] = smoke(model)
        print("Index / restore / export smoke test: PASS")

    finally:
        output = RESULTS / "correctness.json"
        output.write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )

    print("Saved:", RESULTS / "correctness.json")


if __name__ == "__main__":
    main()
