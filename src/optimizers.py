import torch


def get_sgd_optimizer(parameters, learning_rate, weight_decay=0.0):
    """Return SGD without momentum."""
    return torch.optim.SGD(
        parameters,
        lr=learning_rate,
        momentum=0.0,
        weight_decay=weight_decay,
    )


def get_sgd_momentum_optimizer(parameters, learning_rate, momentum=0.9, weight_decay=0.0):
    """Return SGD with momentum."""
    return torch.optim.SGD(
        parameters,
        lr=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
    )


def get_adam_optimizer(parameters, learning_rate, weight_decay=0.0):
    """Return the Adam optimizer."""
    return torch.optim.Adam(
        parameters,
        lr=learning_rate,
        weight_decay=weight_decay,
    )


def get_lbfgs_optimizer(
    parameters,
    learning_rate,
    max_iter=4,
    history_size=10,
):
    """Return L-BFGS.

    L-BFGS needs a closure in the training loop. It is included because this is
    an optimization module project, but it is usually very expensive for large
    neural networks such as Flan-T5.
    """
    return torch.optim.LBFGS(
        parameters,
        lr=learning_rate,
        max_iter=max_iter,
        history_size=history_size,
        line_search_fn="strong_wolfe",
    )


def get_newton_cg_optimizer(*args, **kwargs):
    """Newton-CG is documented as unsupported for this T5 fine-tuning code."""
    raise NotImplementedError(
        "Newton-CG is not implemented for Flan-T5 fine-tuning. "
        "It requires Hessian-vector products and is not provided as a standard "
        "PyTorch optimizer for transformer training."
    )


def get_optimizer(model_parameters, config):
    """Select the optimizer requested in the training configuration."""
    optimizer_name = config.optimizer_name.lower()

    if optimizer_name == "sgd":
        return get_sgd_optimizer(
            model_parameters,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
        )

    if optimizer_name == "sgd_momentum":
        return get_sgd_momentum_optimizer(
            model_parameters,
            learning_rate=config.learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )

    if optimizer_name == "adam":
        return get_adam_optimizer(
            model_parameters,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
        )

    if optimizer_name == "lbfgs":
        return get_lbfgs_optimizer(
            model_parameters,
            learning_rate=config.learning_rate,
            max_iter=config.lbfgs_max_iter,
            history_size=config.lbfgs_history_size,
        )

    if optimizer_name == "newton_cg":
        return get_newton_cg_optimizer()

    raise ValueError(f"Unknown optimizer: {config.optimizer_name}")
