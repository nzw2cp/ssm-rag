
# PRECOG.py
import json
import hashlib
from pathlib import Path

import torch
from tvm_ffi.utils import kwargs_wrapper

# Temporary compatibility workaround for the installed
# CUTLASS DSL / TVM-FFI combination.
if not getattr(kwargs_wrapper, "_precog_patched", False):
    _original_wrapper = kwargs_wrapper.make_kwargs_wrapper

    def _compatible_wrapper(*args, **kwargs):
        kwargs.pop("map_dataclass_to_tuple", None)
        return _original_wrapper(*args, **kwargs)

    kwargs_wrapper.make_kwargs_wrapper = _compatible_wrapper
    kwargs_wrapper._precog_patched = True

from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.utils.generation import InferenceParams


class PRECOG:
    def __init__(
        self,
        model_name="state-spaces/mamba3-siso-1.5b",
        storage_dir="precog_states",
        checkpoint_id=None,
        device="cuda",
        dtype=torch.float16
    ):
        self.model_name = model_name
        self.checkpoint_id = checkpoint_id or model_name
        self.device = device
        self.dtype = dtype
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.storage_dir / "manifest.json"

        print("Loading Mamba 3...")
        self.model = MambaLMHeadModel.from_pretrained(
            model_name, device=device, dtype=dtype
        ).eval()

        self._fix_step_shapes()
        self.manifest = self._load_manifest()
        print("PRECOG initialized successfully")

    def _fix_step_shapes(self):
        # Workaround for the Mamba 3 one-token decoding shape.
        for layer in self.model.backbone.layers:
            mixer = layer.mixer
            original_step = mixer.step

            def compatible_step(
                u, *args, _original=original_step, **kwargs
            ):
                if u.ndim == 3:
                    if u.shape[1] != 1:
                        raise ValueError(
                            "Mamba 3 step requires one token"
                        )
                    result = _original(
                        u.squeeze(1), *args, **kwargs
                    )
                    return (
                        result[0].unsqueeze(1),
                        *result[1:]
                    )
                return _original(u, *args, **kwargs)

            mixer.step = compatible_step

    def _load_manifest(self):
        if self.manifest_path.exists():
            with self.manifest_path.open(
                "r", encoding="utf-8"
            ) as file:
                return json.load(file)
        return {}

    def _save_manifest(self):
        temporary = self.manifest_path.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8") as file:
            json.dump(self.manifest, file, indent=2)
        temporary.replace(self.manifest_path)

    def _state_path(self, chunk_id):
        digest = hashlib.sha256(
            chunk_id.encode("utf-8")
        ).hexdigest()
        return self.storage_dir / f"{digest}.pt"

    def _create_cache(self, max_seqlen):
        return InferenceParams(
            max_seqlen=max_seqlen,
            max_batch_size=1,
            key_value_memory_dict=(
                self.model.allocate_inference_cache(
                    batch_size=1,
                    max_seqlen=max_seqlen,
                    dtype=self.dtype
                )
            )
        )

    def encode_document(self, chunk_id, token_ids):
        """Encode a chunk and persist its recurrent state."""
        if not isinstance(chunk_id, str) or not chunk_id:
            raise ValueError("chunk_id must be a nonempty string")
        if not token_ids:
            raise ValueError("Document cannot be empty")

        tokens = torch.tensor(
            [token_ids], dtype=torch.long, device=self.device
        )
        seq_len = tokens.shape[1]
        cache = self._create_cache(seq_len + 1)

        print(f"Encoding {chunk_id}: {seq_len} tokens")

        with torch.inference_mode():
            self.model(tokens, inference_params=cache)
            states = {
                layer: tuple(
                    t.detach().cpu().clone() for t in tensors
                )
                for layer, tensors
                in cache.key_value_memory_dict.items()
            }

        payload = {
            "model": self.model_name,
            "checkpoint_id": self.checkpoint_id,
            "chunk_id": chunk_id,
            "token_count": seq_len,
            "states": states
        }

        path = self._state_path(chunk_id)
        torch.save(payload, path)

        self.manifest[chunk_id] = {
            "file": path.name,
            "token_count": seq_len,
            "checkpoint_id": self.checkpoint_id
        }
        self._save_manifest()
        return path

    def index_documents(self, documents):
        """Index {chunk_id: token_id_list}."""
        results = {}
        for chunk_id, token_ids in documents.items():
            results[chunk_id] = self.encode_document(
                chunk_id, token_ids
            )
        return results

    def restore_chunk(self, chunk_id, extra_tokens=1024):
        """Inject a saved chunk into a fresh inference cache."""
        if chunk_id not in self.manifest:
            raise KeyError(f"Unknown chunk: {chunk_id}")

        path = self._state_path(chunk_id)
        payload = torch.load(
            path, map_location="cpu", weights_only=True
        )

        if payload["model"] != self.model_name:
            raise ValueError("Model mismatch")
        if payload["checkpoint_id"] != self.checkpoint_id:
            raise ValueError(
                "Checkpoint changed: re-index documents"
            )
        if payload["chunk_id"] != chunk_id:
            raise ValueError("Chunk ID mismatch")

        token_count = payload["token_count"]
        cache = self._create_cache(token_count + extra_tokens)

        saved = payload["states"]
        current = cache.key_value_memory_dict

        if set(saved) != set(current):
            raise ValueError("Cache layer mismatch")

        with torch.no_grad():
            for layer, tensors in saved.items():
                destinations = current[layer]
                if len(tensors) != len(destinations):
                    raise ValueError("Cache tensor count mismatch")

                for dst, src in zip(destinations, tensors):
                    if dst.shape != src.shape:
                        raise ValueError("Cache tensor shape mismatch")
                    dst.copy_(
                        src.to(
                            device=dst.device, dtype=dst.dtype
                        )
                    )

        cache.seqlen_offset = token_count
        return cache

    def continue_from_chunk(
        self, chunk_id, question_token_ids
    ):
        """
        Process question tokens using an injected state.
        Returns logits, not generated answer text.
        """
        if not question_token_ids:
            raise ValueError("Question cannot be empty")

        cache = self.restore_chunk(
            chunk_id,
            extra_tokens=len(question_token_ids) + 1
        )
        question = torch.tensor(
            [question_token_ids],
            dtype=torch.long,
            device=self.device
        )

        logits = []
        with torch.inference_mode():
            for i in range(question.shape[1]):
                output = self.model(
                    question[:, i:i + 1],
                    inference_params=cache
                )
                logits.append(
                    output.logits.detach().float().cpu()
                )
                cache.seqlen_offset += 1

        return torch.cat(logits, dim=1)

    def list_chunks(self):
        return list(self.manifest.keys())

    # HIDDEN STATE EXPORT

    @staticmethod
    def _write_tensor(file, name, tensor, full_values=False):
        """
        Export tensor information to a readable text file.
        """

        tensor = tensor.detach().cpu().contiguous()

        file.write(f"\n{name}\n")
        file.write(f"Shape: {tuple(tensor.shape)}\n")
        file.write(f"Data type: {tensor.dtype}\n")
        file.write(f"Elements: {tensor.numel()}\n")

        flat = tensor.reshape(-1)

        if full_values:
            file.write("All values:\n")

            # Write in small batches to avoid building
            # one enormous Python string.
            for start in range(0, flat.numel(), 256):
                values = flat[start:start + 256].tolist()

                file.write(
                    " ".join(str(v) for v in values)
                    + "\n"
                )
        else:
            file.write("First 16 values:\n")
            file.write(
                str(flat[:16].tolist()) + "\n"
            )
            file.write(
                "[Preview only: set full_values=True "
                "to export every value]\n"
            )

    def export_hidden_states(
        self,
        token_ids,
        output_dir="hidden_state_outputs",
        full_values=False,
        include_intermediate=True,
        include_activations=True
    ):
        """
        Export final and intermediate Mamba 3 states.

        Final states:
            Cache after the complete input.

        Intermediate states:
            Cache after each input-token prefix.

        Layer activations:
            Output tensor from each Mamba layer
            at every input position.
        """

        if not token_ids:
            raise ValueError("token_ids cannot be empty")

        output_dir = Path(output_dir)
        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        tokens = torch.tensor(
            [token_ids],
            dtype=torch.long,
            device=self.device
        )

        # Capture layer activations
        activations = {}
        hooks = []

        if include_activations:

            for layer_id, layer in enumerate(
                self.model.backbone.layers
            ):

                def capture_hook(
                    module,
                    inputs,
                    output,
                    index=layer_id
                ):
                    # Mamba Block returns:
                    # (hidden_states, residual)
                    hidden = output[0]

                    activations[index] = (
                        hidden.detach().cpu().clone()
                    )

                hooks.append(
                    layer.register_forward_hook(
                        capture_hook
                    )
                )

        # Process the complete input
        cache = self._create_cache(
            max_seqlen=len(token_ids) + 1
        )

        try:
            with torch.inference_mode():
                self.model(
                    tokens,
                    inference_params=cache
                )
        finally:
            for hook in hooks:
                hook.remove()

        # Export final recurrent states
        final_path = output_dir / "final_states.txt"

        with final_path.open(
            "w", encoding="utf-8"
        ) as file:

            file.write("MAMBA 3 FINAL RECURRENT STATES\n")
            file.write(f"Model: {self.model_name}\n")
            file.write(f"Tokens: {len(token_ids)}\n")
            file.write(
                f"Layers: "
                f"{len(cache.key_value_memory_dict)}\n"
            )

            for layer_id, states in (
                cache.key_value_memory_dict.items()
            ):

                file.write(
                    f"\n========== LAYER {layer_id} "
                    f"==========\n"
                )

                for state_id, tensor in enumerate(states):

                    self._write_tensor(
                        file,
                        f"Recurrent State {state_id}",
                        tensor,
                        full_values
                    )

        print("Final states saved:", final_path)

        # Export layer activations
        if include_activations:

            activation_path = (
                output_dir / "layer_activations.txt"
            )

            with activation_path.open(
                "w", encoding="utf-8"
            ) as file:

                file.write(
                    "MAMBA 3 LAYER ACTIVATIONS\n"
                )

                for layer_id in sorted(activations):

                    self._write_tensor(
                        file,
                        f"Layer {layer_id} output",
                        activations[layer_id],
                        full_values
                    )

            print(
                "Layer activations saved:",
                activation_path
            )

        #  Export intermediate recurrent states
        if include_intermediate:

            intermediate_dir = (
                output_dir / "intermediate_states"
            )

            intermediate_dir.mkdir(
                parents=True,
                exist_ok=True
            )


            for position in range(1, len(token_ids) + 1):

                prefix = tokens[:, :position]

                prefix_cache = self._create_cache(
                    max_seqlen=position + 1
                )

                with torch.inference_mode():
                    self.model(
                        prefix,
                        inference_params=prefix_cache
                    )

                path = intermediate_dir / (
                    f"token_{position:04d}.txt"
                )

                with path.open(
                    "w", encoding="utf-8"
                ) as file:

                    file.write(
                        "MAMBA 3 INTERMEDIATE "
                        "RECURRENT STATES\n"
                    )

                    file.write(
                        f"After token position: "
                        f"{position}\n"
                    )

                    file.write(
                        f"Token ID: "
                        f"{token_ids[position - 1]}\n"
                    )

                    for layer_id, states in (
                        prefix_cache
                        .key_value_memory_dict.items()
                    ):

                        file.write(
                            f"\n========== LAYER "
                            f"{layer_id} ==========\n"
                        )

                        for state_id, tensor in enumerate(
                            states
                        ):

                            self._write_tensor(
                                file,
                                f"Recurrent State {state_id}",
                                tensor,
                                full_values
                            )

                print(
                    f"Exported intermediate "
                    f"state {position}/{len(token_ids)}"
                )

        print("\nHidden-state export complete!")

        return output_dir
