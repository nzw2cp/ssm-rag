
from pathlib import Path
from PRECOG import PRECOG


def test_hidden_state_export():

    precog = PRECOG()

    # Start with eight synthetic tokens.
    # We will use real document text later.
    token_ids = [1, 2, 3, 4, 5, 6, 7, 8]

    output_folder = precog.export_hidden_states(
        token_ids=token_ids,
        output_dir="hidden_state_outputs",

        # False = first 16 values per tensor
        # True = every numerical value
        full_values=False,

        include_intermediate=True,
        include_activations=True
    )

    # Confirm the output files were created.
    assert (
        output_folder / "final_states.txt"
    ).exists()

    assert (
        output_folder / "layer_activations.txt"
    ).exists()

    intermediate_files = list(
        (output_folder / "intermediate_states")
        .glob("token_*.txt")
    )

    assert len(intermediate_files) == len(token_ids)

    print("\nPASS: Hidden-state export files created")
    print("Output folder:", output_folder.resolve())


if __name__ == "__main__":
    test_hidden_state_export()
