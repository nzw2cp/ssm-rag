
import json
import hashlib
from pathlib import Path

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    MambaCache,
)


class PRECOG:
    def __init__(self, storage_dir="precog_states"):
        self.model_name = "tiiuae/falcon-mamba-7b-instruct"
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.storage_dir / "manifest.json"

        print("Loading Falcon Mamba 7B Instruct...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name
        )

        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map={"": 0},
            quantization_config=quantization,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        ).eval()

        self.device = self.model.get_input_embeddings().weight.device

        self.manifest = (
            json.loads(self.manifest_path.read_text())
            if self.manifest_path.exists() else {}
        )

        print("PRECOG initialized successfully")

    def tokens(self, text_or_ids):
        if isinstance(text_or_ids, str):
            ids = self.tokenizer.encode(
                text_or_ids,
                add_special_tokens=False,
            )
        else:
            ids = list(text_or_ids)

        if not ids:
            raise ValueError("Input cannot be empty")

        return ids

    def forward(
        self,
        ids,
        cache=None,
        position=0,
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
            kwargs["cache_params"] = cache
            kwargs["cache_position"] = torch.arange(
                position,
                position + len(ids),
                device=self.device,
            )

        with torch.inference_mode():
            return self.model(**kwargs)

    def new_cache(self):
        return MambaCache(
            self.model.config,
            max_batch_size=1,
            device=self.device,
            dtype=self.model.dtype,
        )

    @staticmethod
    def snapshot(cache):
        return {
            "conv_states": cache.conv_states.detach().cpu().clone(),
            "ssm_states": cache.ssm_states.detach().cpu().clone(),
        }

    def encode_document(self, chunk_id, text_or_ids):
        ids = self.tokens(text_or_ids)

        outputs = self.forward(ids)
        states = self.snapshot(outputs.cache_params)

        filename = hashlib.sha256(
            chunk_id.encode()
        ).hexdigest() + ".pt"

        torch.save(
            {
                "model": self.model_name,
                "chunk_id": chunk_id,
                "token_count": len(ids),
                "states": states,
            },
            self.storage_dir / filename,
        )

        self.manifest[chunk_id] = {
            "file": filename,
            "token_count": len(ids),
        }

        self.manifest_path.write_text(
            json.dumps(self.manifest, indent=2)
        )

        print(f"Indexed {chunk_id}: {len(ids)} tokens")

    def index_documents(self, documents):
        for chunk_id, content in documents.items():
            self.encode_document(chunk_id, content)

    def list_chunks(self):
        return list(self.manifest)

    def restore_chunk(self, chunk_id):
        entry = self.manifest[chunk_id]

        payload = torch.load(
            self.storage_dir / entry["file"],
            map_location="cpu",
            weights_only=True,
        )

        if (
            payload["model"] != self.model_name
            or payload["chunk_id"] != chunk_id
            or payload["token_count"] != entry["token_count"]
        ):
            raise ValueError("Saved state metadata mismatch")

        cache = self.new_cache()

        with torch.no_grad():
            for name in ("conv_states", "ssm_states"):
                source = payload["states"][name]
                destination = getattr(cache, name)

                if source.shape != destination.shape:
                    raise ValueError(f"{name} shape mismatch")

                destination.copy_(source.to(destination.device))

        return cache

    def continue_from_chunk(self, chunk_id, question):
        ids = self.tokens(question)
        cache = self.restore_chunk(chunk_id)

        position = self.manifest[chunk_id]["token_count"]
        logits = []

        for token_id in ids:
            output = self.forward(
                [token_id],
                cache,
                position,
            )

            logits.append(output.logits.float().cpu())
            position += 1

        return torch.cat(logits, dim=1)

    @staticmethod
    def write_tensor(file, name, tensor, full_values=False):
        tensor = tensor.detach().cpu().contiguous()
        values = tensor.flatten()

        file.write(f"\n{name}\n")
        file.write(f"Shape: {tuple(tensor.shape)}\n")
        file.write(f"Data type: {tensor.dtype}\n")
        file.write(f"Elements: {tensor.numel()}\n")

        if full_values:
            for start in range(0, values.numel(), 256):
                file.write(
                    str(values[start:start + 256].tolist()) + "\n"
                )
        else:
            file.write(
                f"First 16 values: {values[:16].tolist()}\n"
            )

    def write_cache(self, file, cache, full_values=False):
        for name in ("conv_states", "ssm_states"):
            states = getattr(cache, name)

            for layer in range(states.shape[0]):
                self.write_tensor(
                    file,
                    f"Layer {layer} - {name}",
                    states[layer],
                    full_values,
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
        folder.mkdir(parents=True, exist_ok=True)

        outputs = self.forward(
            ids,
            hidden_states=include_activations,
        )

        with (folder / "final_states.txt").open("w") as file:
            file.write("Falcon Mamba final recurrent states\n")
            file.write(f"Tokens: {len(ids)}\n")
            self.write_cache(
                file,
                outputs.cache_params,
                full_values,
            )

        if include_activations:
            with (folder / "layer_activations.txt").open("w") as file:
                file.write("Falcon Mamba layer activations\n")

                for layer, tensor in enumerate(outputs.hidden_states):
                    self.write_tensor(
                        file,
                        f"Layer {layer}",
                        tensor,
                        full_values,
                    )

        if include_intermediate:
            intermediate = folder / "intermediate_states"
            intermediate.mkdir(exist_ok=True)

            cache = None

            for position, token_id in enumerate(ids):
                if cache is None:
                    output = self.forward([token_id])
                    cache = output.cache_params
                else:
                    self.forward([token_id], cache, position)

                path = intermediate / f"token_{position + 1:04d}.txt"

                with path.open("w") as file:
                    file.write(
                        f"Falcon Mamba state after token {position + 1}\n"
                    )
                    file.write(f"Token ID: {token_id}\n")
                    self.write_cache(file, cache, full_values)

        print("Hidden states exported to:", folder.resolve())
        return folder
