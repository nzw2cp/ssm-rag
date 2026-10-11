
# test_precog.py
from PRECOG import PRECOG


def main():
    precog = PRECOG()

    documents = {
        "document-001": list(range(1, 65)),
        "document-002": list(range(100, 164)),
        "document-003": list(range(200, 264))
    }

    print("\n=== INDEX DOCUMENTS ===")
    precog.index_documents(documents)

    print("Indexed chunks:", precog.list_chunks())

    print("\n=== RESTORE A CHUNK ===")
    cache = precog.restore_chunk("document-001")

    print("Restored position:", cache.seqlen_offset)
    print("Cached layers:", len(cache.key_value_memory_dict))

    print("\n=== CONTINUE FROM CACHED STATE ===")
    logits = precog.continue_from_chunk(
        "document-001",
        [65, 66, 67, 68]
    )

    print("Logit shape:", tuple(logits.shape))
    print(
        "Predicted tokens:",
        logits.argmax(dim=-1).tolist()
    )

    assert len(precog.list_chunks()) == 3
    assert cache.seqlen_offset == 64
    assert logits.shape[1] == 4

    print("\nPASS: PRECOG document indexing works!")


if __name__ == "__main__":
    main()
