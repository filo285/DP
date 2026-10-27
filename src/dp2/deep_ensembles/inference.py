"""One deterministic forward pass per independently trained ensemble member."""
import torch
from dp2.common.uncertainty import ProbabilityAccumulator


@torch.inference_mode()
def ensemble_predict(models, images, valid_mask=None, sequential=False):
    models = list(models)
    if len(models) < 2 or len({id(m) for m in models}) != len(models):
        raise ValueError("Provide at least two distinct ensemble models")
    accumulator = ProbabilityAccumulator(valid_mask=valid_mask)
    for model in models:
        modules = list(model.modules())
        states = [m.training for m in modules]
        tensors = list(model.parameters()) + list(model.buffers())
        devices = {tensor.device for tensor in tensors}
        if sequential and len(devices) > 1:
            raise ValueError("Sequential inference requires each member on a single device")
        original_device = next(iter(devices), images.device)
        try:
            if sequential:
                model.to(images.device)
            model.eval()  # Includes dropout AND batch normalization.
            accumulator.update_logits(model(images))
        finally:
            if sequential:
                model.to(original_device)
            for module, state in zip(modules, states):
                module.training = state
    return accumulator.result()
