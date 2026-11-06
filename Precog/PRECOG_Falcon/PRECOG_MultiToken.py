
from contextlib import contextmanager
from types import MethodType
import importlib

import torch


def debug_stage(layer, stage, operation):
    print(f"Layer {layer}: Starting {stage}", flush=True)

    result = operation()

    torch.cuda.synchronize()

    print(f"Layer {layer}: Completed {stage}", flush=True)

    return result


def cached_multi_forward(
    mixer,
    hidden_states,
    cache_params=None,
    cache_position=None,
    attention_mask=None,
):
    fm = importlib.import_module(
        "transformers.models.falcon_mamba.modeling_falcon_mamba"
    )

    if cache_params is None or cache_position is None:
        raise ValueError("A saved cache and position are required")

    if cache_position[0].item() == 0:
        raise ValueError("The document state must already exist")

    layer = mixer.layer_idx

    # Project the continuation tokens together.
    projected = debug_stage(
        layer,
        "Input projection",
        lambda: mixer.in_proj(hidden_states).transpose(1, 2),
    )

    values, gate = projected.chunk(2, dim=1)

    if attention_mask is not None:
        values = values * attention_mask.unsqueeze(1)

    token_count = values.shape[-1]

    previous_conv = cache_params.conv_states[layer]
    state_length = previous_conv.shape[-1]

    # Combine the previous convolution history with new tokens.
    combined = torch.cat(
        [previous_conv.to(values.dtype), values],
        dim=-1,
    ).contiguous()

    conv_weights = mixer.conv1d.weight.view(
        mixer.conv1d.weight.size(0),
        mixer.conv1d.weight.size(2),
    )

    conv_output = debug_stage(
        layer,
        "Convolution",
        lambda: fm.causal_conv1d_fn(
            combined,
            conv_weights,
            mixer.conv1d.bias,
            activation=mixer.activation,
        )[:, :, -token_count:].contiguous(),
    )

    # Save the final convolution history.
    with torch.no_grad():
        previous_conv.copy_(
            combined[:, :, -state_length:]
        )

    if attention_mask is not None:
        conv_output = conv_output * attention_mask.unsqueeze(1)

    ssm_parameters = debug_stage(
        layer,
        "SSM parameter projection",
        lambda: mixer.x_proj(
            conv_output.transpose(1, 2).contiguous()
        ),
    )

    time_step, B, C = torch.split(
        ssm_parameters,
        [
            mixer.time_step_rank,
            mixer.ssm_state_size,
            mixer.ssm_state_size,
        ],
        dim=-1,
    )

    B = fm.rms_forward(
        B,
        variance_epsilon=mixer.rms_eps,
    )

    C = fm.rms_forward(
        C,
        variance_epsilon=mixer.rms_eps,
    )

    time_step = fm.rms_forward(
        time_step,
        variance_epsilon=mixer.rms_eps,
    )

    def project_time_step():
        if hasattr(mixer.config, "_pre_quantization_dtype"):
            return (
                mixer.dt_proj(time_step) - mixer.dt_proj.bias
            ).transpose(1, 2)

        return (
            mixer.dt_proj.weight
            @ time_step.transpose(1, 2)
        )

    discrete_time_step = debug_stage(
        layer,
        "Time-step projection",
        project_time_step,
    )

    A = -torch.exp(mixer.A_log.float())

    time_proj_bias = (
        mixer.dt_proj.bias.float()
        if mixer.dt_proj.bias is not None
        else None
    )

    ssm_state = cache_params.ssm_states[layer]
    scan_outputs = []

    # Update the recurrent state in token order.
    for position in range(token_count):
        token_hidden = conv_output[
            :, :, position
        ].contiguous()

        token_dt = discrete_time_step[
            :, :, position
        ].contiguous()

        token_B = B[
            :, position, :
        ].contiguous()

        token_C = C[
            :, position, :
        ].contiguous()

        token_gate = gate[
            :, :, position
        ].contiguous()

        def update_state():
            return fm.selective_state_update(
                ssm_state,
                token_hidden,
                token_dt,
                A,
                token_B,
                token_C,
                mixer.D,
                token_gate,
                time_proj_bias,
                dt_softplus=True,
            )

        output = debug_stage(
            layer,
            f"SSM update for token {position + 1}",
            update_state,
        )

        scan_outputs.append(output)

    scan_outputs = torch.stack(
        scan_outputs,
        dim=-1,
    )

    result = debug_stage(
        layer,
        "Output projection",
        lambda: mixer.out_proj(
            scan_outputs.transpose(1, 2).contiguous()
        ),
    )

    return result


@contextmanager
def multitoken_context(model):
    original_methods = []

    try:
        for block in model.backbone.layers:
            mixer = block.mixer

            if not hasattr(mixer, "cuda_kernels_forward"):
                raise RuntimeError(
                    "Unsupported Falcon Mamba mixer"
                )

            had_original = (
                "cuda_kernels_forward" in mixer.__dict__
            )

            original = mixer.__dict__.get(
                "cuda_kernels_forward"
            )

            original_methods.append(
                (mixer, had_original, original)
            )

            mixer.cuda_kernels_forward = MethodType(
                cached_multi_forward,
                mixer,
            )

        yield

    finally:
        # Restore the model methods after the experiment.
        for mixer, had_original, original in reversed(
            original_methods
        ):
            if had_original:
                mixer.cuda_kernels_forward = original
            else:
                mixer.__dict__.pop(
                    "cuda_kernels_forward",
                    None,
                )


class PRECOGMultiToken:
    def __init__(self, precog):
        self.precog = precog

    def restore_snapshot(self, snapshot):
        cache = self.precog.new_cache()

        with torch.no_grad():
            for name in ("conv_states", "ssm_states"):
                destination = getattr(cache, name)
                source = snapshot[name]

                if source.shape != destination.shape:
                    raise ValueError(
                        f"Invalid shape for {name}"
                    )

                destination.copy_(
                    source.to(
                        device=destination.device,
                        dtype=destination.dtype,
                    )
                )

        return cache

    def continue_tokens(
        self,
        snapshot,
        context_length,
        tokens,
        method="multitoken",
    ):
        if method not in ("multitoken", "sequential"):
            raise ValueError(
                "Method must be multitoken or sequential"
            )

        ids = self.precog.tokens(tokens)

        if context_length < 1:
            raise ValueError(
                "Context length must be positive"
            )

        cache = self.restore_snapshot(snapshot)

        if method == "sequential":
            outputs = []

            for position, token_id in enumerate(ids):
                result = self.precog.forward(
                    [token_id],
                    cache=cache,
                    position=context_length + position,
                    hidden_states=False,
                )

                outputs.append(
                    result.logits.float().cpu().clone()
                )

            logits = torch.cat(outputs, dim=1)

        else:
            # Use the original method for one token.
            if len(ids) == 1:
                result = self.precog.forward(
                    ids,
                    cache=cache,
                    position=context_length,
                    hidden_states=False,
                )

            else:
                # Use our local implementation for multiple tokens.
                with multitoken_context(self.precog.model):
                    result = self.precog.forward(
                        ids,
                        cache=cache,
                        position=context_length,
                        hidden_states=False,
                    )

            logits = result.logits.float().cpu().clone()

        final_snapshot = self.precog.snapshot(cache)

        return logits, final_snapshot
