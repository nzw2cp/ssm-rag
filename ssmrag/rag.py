# import libraries for testing the setup
import sys
import torch
import hnswlib

# Baseline RAG Model

# Import libraries 
from transformers import AutoTokenizer
from mamba_ssm import MambaLMHeadModel

from typing import Any, List, Optional, Callable, Literal, cast
import numpy as np

from datasets import load_dataset

# @torch.inference_mode() # No need to decorate the class with inference_mode, as it is not a function.
class RAG():
    def __init__(
            self,
            model_id: str,
            tokenizer_id: str,
            device: str,
            dtype: torch.dtype | None,
            space: Literal['l2', 'ip', 'cosine'],
            dim: int,
        ):

        self.index = hnswlib.Index(space, dim)
        self.embeddings = []
        self.chunk_data = []

        self.device = device
        self.dtype  = dtype
        self.space: Literal['l2', 'ip', 'cosine'] = space
        self.dim = dim

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_id,
            token=True,
        )

        self.model = MambaLMHeadModel.from_pretrained(
            model_id,
            device=self.device,
            dtype=self.dtype,
        )
    
    @torch.inference_mode()
    def _baseline(
            self,
            title: str,
            text: str,
            chunk_size: int = 512,
            overlap: float = 0.1
        ):
            token_ids = self.tokenizer(
                text,
                return_tensors="pt",
                add_special_tokens=False
            ).input_ids[0].to(self.device)

            # Validate chunk_size and overlap
            if chunk_size <= 0:
                raise ValueError("chunk_size must be positive.")
            if not 0 <= overlap < 1:
                raise ValueError("overlap must be between 0 and 1.")

            stride = max(1, int(chunk_size * (1 - overlap)))

            for i in range(0, len(token_ids), stride):
                chunk = token_ids[i:i + chunk_size]

                input_ids = chunk.unsqueeze(0)

                emb = self.model.backbone(input_ids)

                state_vector = emb.squeeze(0)
                state_vector = state_vector.float().cpu().numpy()

                self.chunk_data.append({
                    "embedding": state_vector,
                    "title": title,
                    "start": i,
                    "stop": min(i + chunk_size, len(token_ids)),
                    "text": self.tokenizer.decode(
                        chunk,
                        skip_special_tokens=True
                    )
                })


    # Original ingest method commented out for reference or rollback
    # def ingest(self, corpus, strategy, **kwargs):
    #     # 1. Tokenize
    #     token_ids = self.tokenizer(
    #         corpus,
    #         return_tensors="pt",
    #         add_special_tokens=False,
    #     ).input_ids[0].to(self.device)          # shape: (seq_len,)

    #     if strategy == 'baseline':
    #         self._baseline(token_ids, **kwargs)
    #     else:
    #         raise NotImplementedError(f'Strategy {strategy} has not yet been implemented.')

    #     self.index.init_index(max_elements = len(self.chunk_data), ef_construction = 200, M = 16)

    #     self.index.add_items(
    #         data=[c['embedding'] for c in self.chunk_data],
    #         ids=range(len(self.chunk_data)),
    #     )

    # New ingest method that takes a dictionary of documents instead of a single corpus string
    def ingest(self, documents, strategy="baseline", **kwargs):

        self.chunk_data = []

        for title, text in documents.items():

            if strategy == "baseline":
                self._baseline(
                    title=title,
                    text=text,
                    **kwargs
                )
            else:
                raise NotImplementedError(
                    f"Strategy {strategy} has not been implemented."
                )

        if not self.chunk_data:
            raise ValueError("No chunks generated.")

        # Create ONE index for the entire corpus
        self.index = hnswlib.Index(
            space=self.space,
            dim=self.dim
        )

        self.index.init_index(
            max_elements=len(self.chunk_data),
            ef_construction=200,
            M=16
        )

        embeddings = np.vstack([
            chunk["embedding"] for chunk in self.chunk_data
        ]).astype(np.float32)

        self.index.add_items(
            embeddings,
            ids=np.arange(len(self.chunk_data))
        )

        print(f"Documents: {len(documents)}")
        print(f"Chunks: {len(self.chunk_data)}")

    @torch.inference_mode()
    def query(self, question: str, k: int = 3, max_new_tokens: int = 100):

        # 1. Tokenize question
        input_ids = self.tokenizer(
            question,
            return_tensors="pt",
            add_special_tokens=False
        ).input_ids.to(self.device)

        # 2. Generate question embedding
        embeddings = self.model.backbone(input_ids)

        query_embedding = embeddings.mean(dim=1).squeeze(0)
        query_embedding = query_embedding.float().cpu().numpy()

        # 3. Retrieve top-k chunks
        count = self.index.get_current_count()

        if count == 0:
            raise ValueError("The HNSW index is empty.")

        labels, distances = self.index.knn_query(
            query_embedding.reshape(1, -1),
            k=min(k, count)
        )

        # 4. Collect retrieved context and metadata
        retrieved = []
        retrieved_context = []

        for idx, distance in zip(labels[0], distances[0]):

            chunk = self.chunk_data[idx]

            retrieved.append({
                "title": chunk["title"],
                "text": chunk["text"],
                "distance": float(distance)
            })

            retrieved_context.append(
                f"Source: {chunk['title']}\n"
                f"Content: {chunk['text']}"
            )

        context = "\n\n".join(retrieved_context)

        # 5. Construct prompt
        prompt = f"""
        Answer the following question using only the provided context.

        If the context does not contain enough information,
        state that you cannot determine the answer.

        Context:
        {context}

        Question:
        {question}

        Answer:
        """

        # 6. Tokenize prompt
        prompt_ids = self.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=True
        ).input_ids.to(self.device)

        # 7. Generate answer
        generated_tokens = []
        for _ in range(max_new_tokens):
            logits = self.model(
                torch.cat([prompt_ids, *generated_tokens], dim=1)
            ).logits[:, -1, :]
            next_token = logits.argmax(dim=-1, keepdim=True)
            generated_tokens.append(next_token)

            if self.tokenizer.eos_token_id is not None and torch.all(
                next_token == self.tokenizer.eos_token_id
            ):
                break

        # 8. Decode generated response
        generated_tokens = torch.cat(generated_tokens, dim=1)[0]

        answer = self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True
        )

        # 9. Return answer and retrieval information
        return {
            "answer": answer.strip(),
            "retrieved": retrieved
        }

def load_2wiki_corpus(dataset):
    documents = {}

    for example in dataset:
        titles = example["context"]["title"]
        sentences = example["context"]["sentences"]

        for title, sentence_list in zip(titles, sentences):

            # Avoid duplicate Wikipedia documents
            if title not in documents:
                documents[title] = " ".join(sentence_list)

    return documents

def evaluate(rag, eval_dataset, k=3, max_new_tokens=100):

    results = []

    for i, example in enumerate(eval_dataset):

        question = example["question"]
        expected = example["answer"]

        output = rag.query(
            question,
            k=k,
            max_new_tokens=max_new_tokens
        )

        results.append({
            "id": example["id"],
            "question": question,
            "expected": expected,
            "predicted": output["answer"],
            "retrieved": output["retrieved"],
            "supporting_facts": example["supporting_facts"]
        })

        print(f"Completed {i + 1}/{len(eval_dataset)}")

    return results


# Main evaluation pipeline

# Load dataset
dataset = load_dataset(
    "framolfese/2WikiMultihopQA",
    split="validation"
)

# Select evaluation questions
eval_dataset = dataset.shuffle(seed=42).select(range(3)) # Adjust the range for more evaluation questions

# Build full validation corpus
documents = load_2wiki_corpus(dataset)

# Initialize model
rag = RAG(
    model_id="state-spaces/mamba3-siso-1.5b",
    tokenizer_id="meta-llama/Llama-3.1-8B",
    device="cuda",
    dtype=torch.bfloat16,
    space="cosine",
    dim=2048
)

# Ingest corpus
rag.ingest(
    documents,
    strategy="baseline",
    chunk_size=512,
    overlap=0.1
)

# Evaluate
results = evaluate(
    rag,
    eval_dataset,
    k=3,
    max_new_tokens=100
)

# Save results
import json

with open("evaluation_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4, ensure_ascii=False)