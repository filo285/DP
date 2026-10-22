"""Streaming MC-dropout estimates for multiclass segmentation (entropy in nats)."""
from contextlib import contextmanager, nullcontext
from dp2.common.uncertainty import ProbabilityAccumulator

import torch
from torch import nn


@contextmanager
def dropout_inference(model):
    """Enable only dropout; restore every module's original mode even on failure."""
    modules = list(model.modules())
    states = [m.training for m in modules]
    dropout_types = (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d,
                     nn.AlphaDropout, nn.FeatureAlphaDropout)
    if not any(isinstance(m, dropout_types) and 0 < m.p < 1 for m in modules):
        raise ValueError("MC dropout requires an existing dropout layer with 0 < p < 1")
    try:
        model.eval()
        for m in modules:
            if isinstance(m, dropout_types):
                m.train()
        yield
    finally:
        for m, state in zip(modules, states):
            m.training = state


@torch.inference_mode()
def mc_dropout_predict(model, images, passes=30, seed=None, valid_mask=None):
    """Average softmax probabilities, never logits or integer class labels.

    Welford accumulation needs O(B*C*H*W) memory regardless of pass count.
    A seed gives repeatability for a fixed batch/device and preserves caller RNG.
    """
    if isinstance(passes, bool) or not isinstance(passes, int) or passes < 2:
        raise ValueError("passes must be an integer >= 2")
    devices = [images.device.index if images.device.index is not None else torch.cuda.current_device()] if images.is_cuda else []
    rng_context = torch.random.fork_rng(devices=devices) if seed is not None else nullcontext()
    with rng_context, dropout_inference(model):
        if seed is not None:
            torch.random.default_generator.manual_seed(seed)
            for device in devices:
                torch.cuda.default_generators[device].manual_seed(seed)
        accumulator = ProbabilityAccumulator(valid_mask=valid_mask)
        for _ in range(passes):
            accumulator.update_logits(model(images))
        return accumulator.result()
