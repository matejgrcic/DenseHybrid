def parameter_count(module):
    trainable, non_trainable = 0, 0
    for p in module.parameters():
        if p.requires_grad:
            trainable += p.numel()
        else:
            non_trainable += p.numel()
    return trainable, non_trainable