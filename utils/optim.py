"""Parameter grouping only; each experiment chooses its optimizer."""

def weight_decay_groups(model, weight_decay):
    excluded = model.no_weight_decay() if hasattr(model, "no_weight_decay") else set()
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            (no_decay if parameter.ndim == 1 or name in excluded else decay).append(parameter)
    return [dict(params=decay, weight_decay=weight_decay), dict(params=no_decay, weight_decay=0.0)]
