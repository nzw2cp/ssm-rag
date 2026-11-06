
from __future__ import annotations

import hashlib
import json
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from types import MethodType
from typing import Mapping, Sequence

import torch
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    MambaCache,
)
from transformers.models.falcon_mamba import modeling_falcon_mamba as fm


def _multitoken_forward(
    mixer,
    hidden_states,
    cache_params=None,
    cache_position=None,
    attention_mask=None,
):
    # Multi-token processing requires an existing document state.
    if cache_params is None or cache_position is None:
        raise ValueError("Multi-token continuation requires a cache")

    if cache_position[0].item() <= 0:
        raise ValueError("Encode the document before continuation")

    layer = mixer.layer_idx

    # Project all continuation tokens together.
    projected = mixer.in_proj(hidden_states).transpose(1, 2)
    values, gate = projected.chunk(2, dim=1)

    if attention_mask is not None:
        values = values * attention_mask.unsqueeze(1)

    count = values.shape[-1]

    # Combine the previous convolution state with new tokens.
    conv_state = cache_params.conv_states[layer]
    kernel_size = conv_state.shape[-1]

    combined = torch.cat(
        (conv_state.to(values.dtype), values),
        dim=-1,
    ).contiguous()

    weights = mixer.conv1d.weight.view(
        mixer.conv1d.weight.size(0),
        mixer.conv1d.weight.size(2),
    )

    # Process the convolution in one operation.
    conv = fm.causal_conv1d_fn(
        combined,
        weights,
        mixer.conv1d.bias,
        activation=mixer.activation,
    )[:, :, -count:].contiguous()

    # Preserve the newest convolution history.
    conv_state.copy_(
        combined[:, :, -kernel_size:]
    )

    if attention_mask is not None:
        conv = conv * attention_mask.unsqueeze(1)

    # Calculate SSM parameters for all new tokens.
    parameters = mixer.x_proj(
        conv.transpose(1, 2).contiguous()
    )

    dt, B, C = torch.split(
        parameters,
        [
            mixer.time_step_rank,
            mixer.ssm_state_size,
            mixer.ssm_state_size,
        ],
        dim=-1,
    )

    dt = fm.rms_forward(
        dt, variance_epsilon=mixer.rms_eps
    )
    B = fm.rms_forward(
        B, variance_epsilon=mixer.rms_eps
    )
    C = fm.rms_forward(
        C, variance_epsilon=mixer.rms_eps
    )

    if hasattr(mixer.config, "_pre_quantization_dtype"):
        dt = (
            mixer.dt_proj(dt) - mixer.dt_proj.bias
        ).transpose(1, 2)
    else:
        dt = (
            mixer.dt_proj.weight
            @ dt.transpose(1, 2)
        )

    A = -torch.exp(mixer.A_log.float())

    bias = (
        mixer.dt_proj.bias.float()
        if mixer.dt_proj.bias is not None
        else None
    )

    state = cache_params.ssm_states[layer]
    outputs = []

    # Recurrent state updates must remain in token order.
    for i in range(count):
        output = fm.selective_state_update(
            state,
            conv[:, :, i].contiguous(),
            dt[:, :, i].contiguous(),
            A,
            B[:, i, :].contiguous(),
            C[:, i, :].contiguous(),
            mixer.D,
            gate[:, :, i].contiguous(),
            bias,
            dt_softplus=True,
        )

        outputs.append(output)

    # Process the output projection for all tokens together.
    return mixer.out_proj(
        torch.stack(outputs, dim=1).contiguous()
    )


@contextmanager
def _multitoken_mode(model):
    # Temporarily use our local multi-token implementation.
    originals = []

    try:
        for block in model.backbone.layers:
            mixer = block.mixer

            existed = (
                "cuda_kernels_forward" in mixer.__dict__
            )

            original = mixer.__dict__.get(
                "cuda_kernels_forward"
            )

            originals.append(
                (mixer, existed, original)
            )

            mixer.cuda_kernels_forward = MethodType(
                _multitoken_forward,
                mixer,
            )

        yield

    finally:
        # Always restore the model's original methods.
        for mixer, existed, original in reversed(originals):
            if existed:
                mixer.cuda_kernels_forward = original
            else:
                mixer.__dict__.pop(
                    "cuda_kernels_forward",
                    None,
                )


class PRECOG:
    def __init__(
        self,
        model_name="tiiuae/falcon-mamba-7b-instruct",
        storage_dir=None,
        checkpoint_id=None,
    ):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is unavailable. Check the GPU setup."
            )

        if transformers.__version__ != "4.47.1":
            raise RuntimeError(
                "This implementation was tested with "
                "transformers==4.47.1"
            )

        self.model_name = model_name
        self.checkpoint_id = checkpoint_id or model_name

        self.storage_dir = Path(
            storage_dir
            or Path(__file__).resolve().parent / "precog_states"
        )

        self.storage_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.manifest_path = (
            self.storage_dir / "manifest.json"
        )

        self.quantization_name = (
            "bitsandbytes-4bit-nf4-double-fp16"
        )

        # Prevent concurrent calls while methods are temporarily replaced.
        self._model_lock = threading.RLock()

        print("Loading Falcon Mamba 7B Instruct...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name
        )

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
            raise ValueError(
                "Expected a Falcon Mamba model"
            )

        self.device = (
            self.model.get_input_embeddings().weight.device
        )

        self.config_hash = hashlib.sha256(
            self.model.config.to_json_string().encode()
        ).hexdigest()

        self.manifest = self._load_manifest()

        print("PRECOG initialized successfully")

    def tokens(
        self,
        source: str | Sequence[int],
    ) -> list[int]:
        if isinstance(source, str):
            ids = self.tokenizer.encode(
                source,
                add_special_tokens=False,
            )
        else:
            ids = [int(token) for token in source]

        if not ids:
            raise ValueError(
                "Token sequence cannot be empty"
            )

        if min(ids) < 0 or max(ids) >= self.model.config.vocab_size:
            raise ValueError(
                "Token ID outside model vocabulary"
            )

        return ids

    @torch.inference_mode()
    def forward(
        self,
        ids,
        cache=None,
        position=None,
        hidden_states=False,
    ):
        ids = self.tokens(ids)

        kwargs = {
            "input_ids": torch.tensor(
                [ids],
                device=self.device,
                dtype=torch.long,
            ),
            "use_cache": True,
            "return_dict": True,
            "output_hidden_states": hidden_states,
        }

        if cache is not None:
            if position is None:
                position = int(
                    getattr(cache, "seqlen_offset", 0)
                )

            kwargs["cache_params"] = cache

            kwargs["cache_position"] = torch.arange(
                position,
                position + len(ids),
                device=self.device,
            )

        with self._model_lock:
            result = self.model(**kwargs)

            # Track the current position ourselves.
            if cache is not None:
                cache.seqlen_offset = (
                    position + len(ids)
                )
            else:
                returned_cache = getattr(
                    result,
                    "cache_params",
                    None,
                )

                if returned_cache is not None:
                    returned_cache.seqlen_offset = len(ids)

        return result

    def new_cache(self):
        cache = MambaCache(
            self.model.config,
            max_batch_size=1,
            device=self.device,
            dtype=self.model.dtype,
        )

        # Some MambaCache versions do not initialize this field.
        cache.seqlen_offset = 0

        return cache

    @staticmethod
    def snapshot(cache):
        return {
            "conv_states": (
                cache.conv_states.detach().cpu().clone()
            ),
            "ssm_states": (
                cache.ssm_states.detach().cpu().clone()
            ),
            "seqlen_offset": int(
                getattr(cache, "seqlen_offset", 0)
            ),
        }

    def restore_snapshot(self, snapshot):
        cache = self.new_cache()

        with torch.no_grad():
            for name in ("conv_states", "ssm_states"):
                source = snapshot[name]
                target = getattr(cache, name)

                if source.shape != target.shape:
                    raise ValueError(
                        f"{name} shape mismatch"
                    )

                target.copy_(
                    source.to(
                        device=target.device,
                        dtype=target.dtype,
                    )
                )

        cache.seqlen_offset = int(
            snapshot.get("seqlen_offset", 0)
        )

        return cache

    @torch.inference_mode()
    def prepare(self, text_or_ids):
        ids = self.tokens(text_or_ids)

        result = self.forward(ids)

        state = self.snapshot(
            result.cache_params
        )

        state["seqlen_offset"] = len(ids)

        return len(ids), state

    @torch.inference_mode()
    def continue_tokens(
        self,
        snapshot,
        context_length,
        tokens,
        method="multitoken",
    ):
        if method not in ("sequential", "multitoken"):
            raise ValueError(
                "Method must be sequential or multitoken"
            )

        ids = self.tokens(tokens)

        if context_length < 1:
            raise ValueError(
                "Context length must be positive"
            )

        cache = self.restore_snapshot(snapshot)
        cache.seqlen_offset = context_length

        with self._model_lock:
            if method == "sequential":
                outputs = []

                for i, token in enumerate(ids):
                    result = self.forward(
                        [token],
                        cache=cache,
                        position=context_length + i,
                    )

                    outputs.append(
                        result.logits.float().cpu().clone()
                    )

                logits = torch.cat(
                    outputs,
                    dim=1,
                )

            elif len(ids) == 1:
                # Use Falcon Mamba's existing single-token path.
                result = self.forward(
                    ids,
                    cache=cache,
                    position=context_length,
                )

                logits = (
                    result.logits.float().cpu().clone()
                )

            else:
                # Use the optimized multi-token path.
                with _multitoken_mode(self.model):
                    result = self.forward(
                        ids,
                        cache=cache,
                        position=context_length,
                    )

                    logits = (
                        result.logits.float().cpu().clone()
                    )

            final_state = self.snapshot(cache)

        return logits, final_state

    def _load_manifest(self):
        if not self.manifest_path.exists():
            return {}

        return json.loads(
            self.manifest_path.read_text(
                encoding="utf-8"
            )
        )

    def _save_manifest(self):
        path = self.manifest_path.with_suffix(
            ".tmp"
        )

        path.write_text(
            json.dumps(self.manifest, indent=2),
            encoding="utf-8",
        )

        path.replace(
            self.manifest_path
        )

    def _state_path(self, chunk_id):
        filename = hashlib.sha256(
            chunk_id.encode()
        ).hexdigest()

        return self.storage_dir / f"{filename}.pt"

    def encode_document(
        self,
        chunk_id,
        text_or_ids,
    ):
        if not isinstance(chunk_id, str) or not chunk_id:
            raise ValueError(
                "chunk_id must be nonempty"
            )

        length, state = self.prepare(
            text_or_ids
        )

        payload = {
            "model": self.model_name,
            "checkpoint_id": self.checkpoint_id,
            "config_hash": self.config_hash,
            "transformers_version": transformers.__version__,
            "quantization": self.quantization_name,
            "chunk_id": chunk_id,
            "token_count": length,
            "states": state,
        }

        path = self._state_path(
            chunk_id
        )

        torch.save(
            payload,
            path,
        )

        self.manifest[chunk_id] = {
            "file": path.name,
            "token_count": length,
            "checkpoint_id": self.checkpoint_id,
        }

        self._save_manifest()

        return path

    def index_documents(
        self,
        documents: Mapping,
    ):
        return {
            chunk_id: self.encode_document(
                chunk_id,
                text,
            )
            for chunk_id, text in documents.items()
        }

    def list_chunks(self):
        return list(
            self.manifest
        )

    def restore_chunk(
        self,
        chunk_id,
        extra_tokens=1024,
    ):
        if chunk_id not in self.manifest:
            raise KeyError(
                f"Unknown chunk: {chunk_id}"
            )

        payload = torch.load(
            self._state_path(chunk_id),
            map_location="cpu",
            weights_only=True,
        )

        expected = {
            "model": self.model_name,
            "checkpoint_id": self.checkpoint_id,
            "config_hash": self.config_hash,
            "transformers_version": transformers.__version__,
            "quantization": self.quantization_name,
            "chunk_id": chunk_id,
        }

        for name, value in expected.items():
            if payload.get(name) != value:
                raise ValueError(
                    f"Saved state {name} mismatch. "
                    "Re-index the document."
                )

        if (
            payload["states"].get("seqlen_offset")
            != payload["token_count"]
        ):
            raise ValueError(
                "Saved document position does not match token count"
            )

        return self.restore_snapshot(
            payload["states"]
        )

    def continue_from_chunk(
        self,
        chunk_id,
        question,
        method="multitoken",
    ):
        cache = self.restore_chunk(chunk_id)
        state = self.snapshot(cache)

        logits, _ = self.continue_tokens(
            state,
            cache.seqlen_offset,
            question,
            method=method,
        )

        return logits

    @staticmethod
    def tensor_memory_mb(tensor):
        return (
            tensor.numel()
            * tensor.element_size()
            / 1024**2
        )

    @torch.inference_mode()
    def extract_hidden_states(self, text_or_ids):
        ids = self.tokens(text_or_ids)

        torch.cuda.synchronize()
        baseline = torch.cuda.memory_allocated()

        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()

        result = self.forward(
            ids,
            hidden_states=True,
        )

        torch.cuda.synchronize()

        return {
            "hidden_states": result.hidden_states,
            "cache": result.cache_params,
            "token_count": len(ids),
            "peak_gpu_memory": (
                torch.cuda.max_memory_allocated()
            ),
            "baseline_gpu_memory": baseline,
            "elapsed_seconds": (
                time.perf_counter() - start
            ),
        }

    def print_hidden_state_summary(self, results):
        states = results["hidden_states"]

        print("\nHidden-state summary")

        for i, tensor in enumerate(states):
            print(
                f"Layer {i}: "
                f"{tuple(tensor.shape)}, "
                f"{self.tensor_memory_mb(tensor):.3f} MiB"
            )

        print(
            "Forward time:",
            f"{results['elapsed_seconds']:.4f}s",
        )

        print(
            "Cache position:",
            getattr(
                results["cache"],
                "seqlen_offset",
                0,
            ),
        )

    @staticmethod
    def _write_tensor(
        file,
        name,
        tensor,
        full_values=False,
    ):
        tensor = tensor.detach().cpu().contiguous()

        file.write(
            f"\n{name}\n"
            f"Shape: {tuple(tensor.shape)}\n"
            f"Dtype: {tensor.dtype}\n"
            f"Elements: {tensor.numel()}\n"
        )

        values = tensor.reshape(-1)

        if full_values:
            for part in values.split(256):
                file.write(
                    f"{part.tolist()}\n"
                )
        else:
            file.write(
                f"First 16 values: "
                f"{values[:16].tolist()}\n"
            )

    def export_hidden_states(
        self,
        token_ids,
        output_dir="hidden_state_outputs",
        full_values=False,
        include_intermediate=True,
        include_activations=True,
    ):
        ids = self.tokens(token_ids)

        folder = Path(output_dir)
        folder.mkdir(
            parents=True,
            exist_ok=True,
        )

        output = self.forward(
            ids,
            hidden_states=include_activations,
        )

        final_state = self.snapshot(
            output.cache_params
        )

        final_state["seqlen_offset"] = len(ids)

        final_path = folder / "final_states.txt"

        with final_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            file.write(
                f"Model: {self.model_name}\n"
                f"Tokens: {len(ids)}\n"
            )

            for name in ("conv_states", "ssm_states"):
                tensor = final_state[name]

                for layer in range(tensor.shape[0]):
                    self._write_tensor(
                        file,
                        f"Layer {layer} {name}",
                        tensor[layer],
                        full_values,
                    )

        if include_activations:
            path = folder / "layer_activations.txt"

            with path.open(
                "w",
                encoding="utf-8",
            ) as file:
                for i, tensor in enumerate(
                    output.hidden_states
                ):
                    self._write_tensor(
                        file,
                        f"Layer {i} activation",
                        tensor,
                        full_values,
                    )

        if include_intermediate:
            directory = (
                folder / "intermediate_states"
            )

            directory.mkdir(
                exist_ok=True
            )

            cache = self.new_cache()

            for position, token in enumerate(ids):
                self.forward(
                    [token],
                    cache=cache,
                    position=position,
                )

                state = self.snapshot(cache)

                path = directory / (
                    f"token_{position + 1:04d}.txt"
                )

                with path.open(
                    "w",
                    encoding="utf-8",
                ) as file:
                    file.write(
                        f"Token position: {position + 1}\n"
                        f"Token ID: {token}\n"
                    )

                    for name in (
                        "conv_states",
                        "ssm_states",
                    ):
                        tensor = state[name]

                        for layer in range(
                            tensor.shape[0]
                        ):
                            self._write_tensor(
                                file,
                                f"Layer {layer} {name}",
                                tensor[layer],
                                full_values,
                            )

        return folder

    # Compatibility methods for earlier project scripts.
    _token_ids = tokens
    _forward = forward
    _make_empty_cache = new_cache
    _snapshot = snapshot
    _inject = restore_snapshot
