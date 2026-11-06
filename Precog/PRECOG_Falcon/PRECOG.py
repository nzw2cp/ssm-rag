
"""Falcon Mamba PRECOG: persistent recurrent states and cached multi-token continuation.

Validated target: Transformers 4.47.1, 4-bit NF4 Falcon Mamba, CUDA.
Multi-token continuation is approximate under reduced-precision projections.
"""
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
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, MambaCache
from transformers.models.falcon_mamba import modeling_falcon_mamba as fm


def _cached_multi(mixer, hidden_states, cache_params=None, cache_position=None,
                  attention_mask=None):
    """Batch convolution/projections; retain ordered SSM state updates."""
    if cache_params is None or cache_position is None or cache_position[0].item() < 1:
        raise ValueError("Multi-token continuation requires a prefilled cache")

    layer = mixer.layer_idx
    values, gate = mixer.in_proj(hidden_states).transpose(1, 2).chunk(2, dim=1)
    if attention_mask is not None:
        values = values * attention_mask.unsqueeze(1)

    previous = cache_params.conv_states[layer]
    combined = torch.cat((previous.to(values.dtype), values), dim=-1).contiguous()
    weights = mixer.conv1d.weight.view(mixer.conv1d.weight.size(0), -1)
    conv = fm.causal_conv1d_fn(
        combined, weights, mixer.conv1d.bias, activation=mixer.activation
    )[:, :, -values.shape[-1]:].contiguous()
    previous.copy_(combined[:, :, -previous.shape[-1]:])
    if attention_mask is not None:
        conv = conv * attention_mask.unsqueeze(1)

    projected = mixer.x_proj(conv.transpose(1, 2).contiguous())
    dt, B, C = torch.split(projected, [mixer.time_step_rank,
                                     mixer.ssm_state_size, mixer.ssm_state_size], dim=-1)
    dt, B, C = (fm.rms_forward(x, variance_epsilon=mixer.rms_eps)
                for x in (dt, B, C))
    if hasattr(mixer.config, "_pre_quantization_dtype"):
        dt = (mixer.dt_proj(dt) - mixer.dt_proj.bias).transpose(1, 2)
    else:
        dt = mixer.dt_proj.weight @ dt.transpose(1, 2)

    state = cache_params.ssm_states[layer]
    A = -torch.exp(mixer.A_log.float())
    bias = mixer.dt_proj.bias.float() if mixer.dt_proj.bias is not None else None
    outputs = [fm.selective_state_update(
        state, conv[:, :, i].contiguous(), dt[:, :, i].contiguous(), A,
        B[:, i, :].contiguous(), C[:, i, :].contiguous(), mixer.D,
        gate[:, :, i].contiguous(), bias, dt_softplus=True
    ) for i in range(values.shape[-1])]
    return mixer.out_proj(torch.stack(outputs, dim=1).contiguous())


@contextmanager
def _batched_mode(model):
    """Override only this model instance and always restore the original methods."""
    old = []
    try:
        for block in model.backbone.layers:
            mixer = block.mixer
            old.append((mixer, "cuda_kernels_forward" in mixer.__dict__,
                        mixer.__dict__.get("cuda_kernels_forward")))
            mixer.cuda_kernels_forward = MethodType(_cached_multi, mixer)
        yield
    finally:
        for mixer, existed, method in reversed(old):
            if existed:
                mixer.cuda_kernels_forward = method
            else:
                mixer.__dict__.pop("cuda_kernels_forward", None)


class PRECOG:
    def __init__(self, model_name="tiiuae/falcon-mamba-7b-instruct",
                 storage_dir=None, checkpoint_id=None):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU required")
        if transformers.__version__ != "4.47.1":
            raise RuntimeError("This implementation was tested with transformers==4.47.1")

        self.model_name = model_name
        self.checkpoint_id = checkpoint_id or model_name
        self.storage_dir = Path(storage_dir) if storage_dir is not None else Path(__file__).resolve().parent / "precog_states"
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.storage_dir / "manifest.json"
        self.quantization_name = "bitsandbytes-4bit-nf4-double-fp16"
        self._model_lock = threading.RLock()

        print("Loading Falcon Mamba 7B Instruct...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                          bnb_4bit_compute_dtype=torch.float16,
                                          bnb_4bit_use_double_quant=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map={"": 0}, quantization_config=quantization,
            torch_dtype=torch.float16, low_cpu_mem_usage=True
        ).eval()
        if self.model.config.model_type != "falcon_mamba":
            raise ValueError("Expected Falcon Mamba model")
        self.device = self.model.get_input_embeddings().weight.device
        self.config_hash = hashlib.sha256(self.model.config.to_json_string().encode()).hexdigest()
        self.manifest = json.loads(self.manifest_path.read_text()) if self.manifest_path.exists() else {}
        print("PRECOG initialized successfully")

    def tokens(self, source: str | Sequence[int]) -> list[int]:
        ids = (self.tokenizer.encode(source, add_special_tokens=False) if isinstance(source, str)
               else [int(x) for x in source])
        if not ids or min(ids) < 0 or max(ids) >= self.model.config.vocab_size:
            raise ValueError("Empty sequence or token outside model vocabulary")
        return ids

    @torch.inference_mode()
    def forward(self, ids, cache=None, position=None, hidden_states=False):
        ids = self.tokens(ids)
        options = dict(input_ids=torch.tensor([ids], dtype=torch.long, device=self.device),
                       use_cache=True, return_dict=True, output_hidden_states=hidden_states)
        if cache is not None:
            position = int(getattr(cache, "seqlen_offset", 0)) if position is None else int(position)
            options.update(cache_params=cache, cache_position=torch.arange(
                position, position + len(ids), device=self.device))
        with self._model_lock:
            result = self.model(**options)
            target = cache if cache is not None else getattr(result, "cache_params", None)
            if target is not None:
                target.seqlen_offset = position + len(ids) if cache is not None else len(ids)
        return result

    def new_cache(self):
        cache = MambaCache(self.model.config, max_batch_size=1,
                           device=self.device, dtype=self.model.dtype)
        cache.seqlen_offset = 0
        return cache

    @staticmethod
    def snapshot(cache):
        return {**{name: getattr(cache, name).detach().cpu().clone()
                   for name in ("conv_states", "ssm_states")},
                "seqlen_offset": int(getattr(cache, "seqlen_offset", 0))}

    def restore_snapshot(self, state):
        cache = self.new_cache()
        with torch.no_grad():
            for name in ("conv_states", "ssm_states"):
                dst, src = getattr(cache, name), state[name]
                if dst.shape != src.shape:
                    raise ValueError(f"{name} shape mismatch: {src.shape} vs {dst.shape}")
                dst.copy_(src.to(device=dst.device, dtype=dst.dtype))
        cache.seqlen_offset = int(state.get("seqlen_offset", 0))
        return cache

    @torch.inference_mode()
    def prepare(self, text_or_ids):
        ids = self.tokens(text_or_ids)
        state = self.snapshot(self.forward(ids).cache_params)
        state["seqlen_offset"] = len(ids)
        return len(ids), state

    @torch.inference_mode()
    def continue_tokens(self, snapshot, context_length, tokens, method="multitoken"):
        if method not in {"sequential", "multitoken"}:
            raise ValueError("method must be sequential or multitoken")
        if context_length < 1 or int(snapshot.get("seqlen_offset", context_length)) != context_length:
            raise ValueError("Invalid or mismatched context length")
        ids, cache = self.tokens(tokens), self.restore_snapshot(snapshot)
        cache.seqlen_offset = context_length

        with self._model_lock:
            if method == "sequential":
                logits = torch.cat([
                    self.forward([token], cache=cache, position=context_length + i)
                    .logits.float().cpu().clone()
                    for i, token in enumerate(ids)
                ], dim=1)
            else:
                if len(ids) == 1:
                    result = self.forward(ids, cache=cache, position=context_length)
                else:
                    with _batched_mode(self.model):
                        result = self.forward(ids, cache=cache, position=context_length)
                logits = result.logits.float().cpu().clone()
            return logits, self.snapshot(cache)

    def _state_path(self, chunk_id):
        return self.storage_dir / (hashlib.sha256(chunk_id.encode()).hexdigest() + ".pt")

    def encode_document(self, chunk_id, text_or_ids):
        if not isinstance(chunk_id, str) or not chunk_id:
            raise ValueError("chunk_id must be a nonempty string")
        count, state = self.prepare(text_or_ids)
        payload = dict(model=self.model_name, checkpoint_id=self.checkpoint_id,
                       config_hash=self.config_hash, transformers_version=transformers.__version__,
                       quantization=self.quantization_name, chunk_id=chunk_id,
                       token_count=count, states=state)
        path = self._state_path(chunk_id)
        torch.save(payload, path)
        self.manifest[chunk_id] = dict(file=path.name, token_count=count,
                                       checkpoint_id=self.checkpoint_id)
        temporary = self.manifest_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.manifest, indent=2), encoding="utf-8")
        temporary.replace(self.manifest_path)
        return path

    def index_documents(self, documents: Mapping):
        return {key: self.encode_document(key, text) for key, text in documents.items()}

    def list_chunks(self):
        return list(self.manifest)

    def restore_chunk(self, chunk_id, extra_tokens=1024):
        if chunk_id not in self.manifest:
            raise KeyError(f"Unknown chunk: {chunk_id}")
        payload = torch.load(self._state_path(chunk_id), map_location="cpu", weights_only=True)
        expected = dict(model=self.model_name, checkpoint_id=self.checkpoint_id,
                        config_hash=self.config_hash, transformers_version=transformers.__version__,
                        quantization=self.quantization_name, chunk_id=chunk_id)
        for key, value in expected.items():
            if payload.get(key) != value:
                raise ValueError(f"Saved state {key} mismatch; re-index document")
        if payload["states"]["seqlen_offset"] != payload["token_count"]:
            raise ValueError("Saved document position mismatch")
        return self.restore_snapshot(payload["states"])

    def continue_from_chunk(self, chunk_id, question, method="multitoken"):
        cache = self.restore_chunk(chunk_id)
        return self.continue_tokens(self.snapshot(cache), cache.seqlen_offset,
                                    question, method=method)[0]

    @staticmethod
    def tensor_memory_mb(tensor):
        return tensor.numel() * tensor.element_size() / 1024**2

    @torch.inference_mode()
    def extract_hidden_states(self, text_or_ids):
        ids = self.tokens(text_or_ids)
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        result = self.forward(ids, hidden_states=True)
        torch.cuda.synchronize()
        return dict(hidden_states=result.hidden_states, cache=result.cache_params,
                    token_count=len(ids), peak_gpu_memory=torch.cuda.max_memory_allocated(),
                    baseline_gpu_memory=before, elapsed_seconds=time.perf_counter() - start)

    def print_hidden_state_summary(self, results):
        for i, tensor in enumerate(results["hidden_states"]):
            print(f"Layer {i}: {tuple(tensor.shape)}, {self.tensor_memory_mb(tensor):.3f} MiB")
        print(f"Forward time: {results['elapsed_seconds']:.4f}s")
        print("Cache position:", getattr(results["cache"], "seqlen_offset", 0))

    @staticmethod
    def _write_tensor(file, name, tensor, full_values=False):
        tensor = tensor.detach().cpu().contiguous()
        file.write(f"\n{name}\nShape: {tuple(tensor.shape)}\nDtype: {tensor.dtype}\nElements: {tensor.numel()}\n")
        flat = tensor.reshape(-1)
        if full_values:
            for part in flat.split(256):
                file.write(" ".join(map(str, part.tolist())) + "\n")
        else:
            file.write(f"First 16 values: {flat[:16].tolist()}\n")

    def _write_cache(self, file, state, full_values):
        file.write(f"Sequence offset: {state['seqlen_offset']}\n")
        for name in ("conv_states", "ssm_states"):
            for layer, tensor in enumerate(state[name]):
                self._write_tensor(file, f"Layer {layer} {name}", tensor, full_values)

    @torch.inference_mode()
    def export_hidden_states(self, token_ids, output_dir="hidden_state_outputs",
                             full_values=False, include_intermediate=True,
                             include_activations=True):
        ids = self.tokens(token_ids)
        folder = Path(output_dir)
        folder.mkdir(parents=True, exist_ok=True)
        result = self.forward(ids, hidden_states=include_activations)
        state = self.snapshot(result.cache_params)
        state["seqlen_offset"] = len(ids)
        with (folder / "final_states.txt").open("w", encoding="utf-8") as f:
            self._write_cache(f, state, full_values)
        if include_activations:
            with (folder / "layer_activations.txt").open("w", encoding="utf-8") as f:
                for i, t in enumerate(result.hidden_states):
                    self._write_tensor(f, f"Layer {i} activation", t, full_values)
        if include_intermediate:
            interim = folder / "intermediate_states"
            interim.mkdir(exist_ok=True)
            cache = self.new_cache()
            for i, token in enumerate(ids):
                self.forward([token], cache=cache, position=i)
                with (interim / f"token_{i + 1:04d}.txt").open("w", encoding="utf-8") as f:
                    f.write(f"Token position: {i + 1}\nToken ID: {token}\n")
                    self._write_cache(f, self.snapshot(cache), full_values)
        return folder

    # Compatibility with older scripts that use the previous method names.
    def _forward(self, token_ids, cache=None, hidden_states=False):
        return self.forward(token_ids, cache=cache, hidden_states=hidden_states)

    _token_ids = tokens
    _make_empty_cache = new_cache
    _snapshot = snapshot
    _inject = restore_snapshot
