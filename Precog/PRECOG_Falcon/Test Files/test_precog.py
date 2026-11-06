
from pathlib import Path

from PRECOG import PRECOG


def main():
    base_dir = Path(__file__).resolve().parent

    precog = PRECOG(
        storage_dir=base_dir / "precog_states"
    )

    documents = {
        "document-001": (
            "France is located in Europe. "
            "The capital of France is Paris."
        ),
        "document-002": (
            "Germany is a country in central Europe. "
            "Its capital is Berlin."
        ),
        "document-003": (
            "Japan is an island country in East Asia. "
            "Its capital is Tokyo."
        ),
    }

    print("\nIndexing documents...")
    precog.index_documents(documents)

    print("Indexed chunks:", precog.list_chunks())

    print("\nRestoring document state...")
    cache = precog.restore_chunk("document-001")

    print(
        "Convolution states:",
        tuple(cache.conv_states.shape)
    )
    print(
        "SSM states:",
        tuple(cache.ssm_states.shape)
    )

    print("\nContinuing from saved state...")
    logits = precog.continue_from_chunk(
        "document-001",
        " What is the capital of France?"
    )

    print("Logit shape:", tuple(logits.shape))
    print(
        "Predicted tokens:",
        logits.argmax(dim=-1).tolist()
    )

    print("\nExporting hidden states...")

    output_folder = precog.export_hidden_states(
        token_ids="France is located in Europe.",
        output_dir=base_dir / "hidden_state_outputs",
        full_values=False,
        include_intermediate=True,
        include_activations=True,
    )

    print("Output folder:", output_folder.resolve())
    print("\nPRECOG complete")


if __name__ == "__main__":
    main()
