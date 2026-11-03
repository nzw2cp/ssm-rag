# import libraries for testing the setup
import sys
import torch
import hnswlib

# Baseline RAG Model

# Import libraries 
from transformers import AutoTokenizer
from mamba_ssm import MambaLMHeadModel
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

from typing import List, Optional, Callable, Literal
import numpy as np


@torch.inference_mode()
class RAG():
    def __init__(
            self,
            model_id: str,
            tokenizer_id: str,
            device: str,
            dtype: None,
            space: Literal['l2', 'ip', 'cosine'],
            dim: int,
        ):

        self.index = hnswlib.Index(space, dim)
        self.embeddings = []
        self.chunk_data = []

        self.device = device
        self.dtype  = dtype

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_id,
            token=True,
        )

        self.model = MambaLMHeadModel.from_pretrained(
            model_id,
            device=self.device,
            dtype=self.dtype,
        )

    def _baseline(self, token_ids, chunk_size: int = 512, overlap: float = .1):
        # 3. Loop: while corpus is not empty
        for i in range(0, len(token_ids), int(chunk_size*overlap)):
            chunk = token_ids[i : i + chunk_size]
            input_ids = chunk.unsqueeze(0)          # (1, chunk_len)

            # Forward pass – this updates the internal recurrent state
            outputs = self.model(input_ids)

            with torch.no_grad():
                emb = self.model.backbone.embedding(input_ids)     # (1, L, d)
                state_vector = emb.mean(dim=1).squeeze(0)          # (d,)
                state_vector = state_vector.float().cpu().numpy()

            self.chunk_data.append({
                "embedding": state_vector, 
                "start": i, 
                "stop": i + chunk_size, 
                "text": self.tokenizer.decode(chunk, skip_special_tokens=True)
            })

    def ingest(self, corpus, strategy, **kwargs):
        # 1. Tokenize
        token_ids = self.tokenizer(
            corpus,
            return_tensors="pt",
            add_special_tokens=False,
        ).input_ids[0].to(self.device)          # shape: (seq_len,)

        if strategy == 'baseline':
            self._baseline(token_ids, **kwargs)
        else:
            raise NotImplementedError(f'Strategy {strategy} has not yet been implemented.')

        self.index.init_index(max_elements = len(self.chunk_data), ef_construction = 200, M = 16)

        self.index.add_items(
            data=[c['embedding'] for c in self.chunk_data],
            ids=range(len(self.chunk_data)),
        )

    def query (self):  
        pass

