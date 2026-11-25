
# Standalone Falcon Mamba document-state souping implementation.

import threading

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    MambaCache,
)


class Souping:
    def __init__(self, model_name="tiiuae/falcon-mamba-7b-instruct"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for Falcon Mamba")

        self.model_name = model_name
        self.lock = threading.RLock()

        print("Loading Falcon Mamba for souping...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map={"": 0},
            quantization_config=quantization,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        ).eval()

        if self.model.config.model_type != "falcon_mamba":
            raise ValueError("This implementation requires Falcon Mamba")

        self.device = self.model.get_input_embeddings().weight.device
        print("Souping model initialized successfully")

    def tokens(self, text_or_ids):
        if isinstance(text_or_ids, str):
            ids = self.tokenizer.encode(
                text_or_ids, add_special_tokens=False
            )
        else:
            ids = [int(token) for token in text_or_ids]

        if not ids:
            raise ValueError("Input cannot be empty")

        if min(ids) < 0 or max(ids) >= self.model.config.vocab_size:
            raise ValueError("Token ID outside model vocabulary")

        return ids

    def new_cache(self):
        cache = MambaCache(
            self.model.config,
            max_batch_size=1,
            device=self.device,
            dtype=self.model.dtype,
        )
        cache.seqlen_offset = 0
        return cache

    @staticmethod
    def snapshot(cache, position):
        return {
            "conv_states": cache.conv_states.detach().cpu().clone(),
            "ssm_states": cache.ssm_states.detach().cpu().clone(),
            "seqlen_offset": int(position),
        }

    def restore(self, state):
        cache = self.new_cache()

        with torch.no_grad():
            for name in ("conv_states", "ssm_states"):
                source = state[name]
                target = getattr(cache, name)

                if source.shape != target.shape:
                    raise ValueError(f"Invalid {name} shape")

                target.copy_(
                    source.to(
                        device=target.device,
                        dtype=target.dtype,
                    )
                )

        cache.seqlen_offset = int(state["seqlen_offset"])
        return cache

    @torch.inference_mode()
    def forward(self, ids, cache=None, position=None):
        ids = self.tokens(ids)

        args = {
            "input_ids": torch.tensor(
                [ids], device=self.device, dtype=torch.long
            ),
            "use_cache": True,
            "return_dict": True,
        }

        if cache is not None:
            position = (
                cache.seqlen_offset if position is None else position
            )
            args["cache_params"] = cache
            args["cache_position"] = torch.arange(
                position, position + len(ids), device=self.device
            )

        with self.lock:
            output = self.model(**args)

            if cache is not None:
                cache.seqlen_offset = position + len(ids)

        return output

    @torch.inference_mode()
    def encode(self, document):
        # Independently encode a single document and retain its final state.
        ids = self.tokens(document)
        input_ids = torch.tensor([ids], device=self.device, dtype=torch.long)
        with self.lock:
            output = self.model.backbone(
                input_ids=input_ids, use_cache=True, return_dict=True
            )
        state = self.snapshot(output.cache_params, len(ids))
        state["last_hidden_state"] = output.last_hidden_state.detach().cpu().clone()
        return state

    def encode_documents(self, documents):
        if not documents:
            raise ValueError("At least one document is required")
        return [self.encode(document) for document in documents]

    @staticmethod
    def merge(states, method="average"):
        if not states:
            raise ValueError("No document states provided")

        if method not in ("average", "sum", "max"):
            raise ValueError("Unknown pooling method")

        lengths = [int(state["seqlen_offset"]) for state in states]

        if min(lengths) < 1:
            raise ValueError("Document states must not be empty")

        # A souped state has no true concatenated token position.
        # Use the longest source document as a continuation convention.
        merged = {"seqlen_offset": max(lengths)}

        for name in ("conv_states", "ssm_states"):
            tensors = [state[name] for state in states]
            template = tensors[0]

            if any(t.shape != template.shape for t in tensors):
                raise ValueError(f"Incompatible {name} tensors")

            if len(tensors) == 1:
                merged[name] = template.clone()
                continue

            values = torch.stack(
                [tensor.float() for tensor in tensors], dim=0
            )

            if method == "average":
                combined = values.mean(dim=0)
            elif method == "sum":
                combined = values.sum(dim=0)
            else:
                combined = values.max(dim=0).values

            if not torch.isfinite(combined).all():
                raise ValueError(f"Non-finite values after {name} pooling")

            merged[name] = combined.to(template.dtype)

        return merged

    @torch.inference_mode()
    def answer(self, state, question, max_new_tokens=40):
        # Query processing is intentionally sequential and independent
        # of PRECOG's optimized multi-token implementation.
        cache = self.restore(state)
        question_ids = self.tokens(question)
        logits = None

        for token in question_ids:
            output = self.forward([token], cache=cache)
            logits = output.logits[:, -1, :]

        generated = []
        eos = self.tokenizer.eos_token_id

        for _ in range(max_new_tokens):
            token = int(logits.argmax(dim=-1).item())

            if token == eos:
                break

            generated.append(token)
            output = self.forward([token], cache=cache)
            logits = output.logits[:, -1, :]

        text = self.tokenizer.decode(
            generated, skip_special_tokens=True
        ).strip()

        return text

    @staticmethod
    def state_size_mib(state):
        return sum(
            state[name].numel() * state[name].element_size()
            for name in ("conv_states", "ssm_states")
        ) / 1024**2
