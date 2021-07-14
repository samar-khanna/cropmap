import torch
import torch.nn as nn
from typing import Callable, Any, Optional


def apply_to_model_parameters(
        model: nn.Module,
        param_func: Callable[[Any], Any],
        module_func: Optional[Callable[[Any], Any]] = None,
        param_check_func: Callable[[Any], bool] = lambda p: True,
        memo_key_func: Optional[Callable[[Any], Any]] = None,
        memo=None
) -> nn.Module:
    """
    Recursively applies a function to each parameter of a nn.Module model such that
    the new model's parameters are linked to the original in the computational graph.
    Useful for MAML ops.
    Inspiration: https://github.com/learnables/learn2learn/blob/master/learn2learn/utils.py#L51
    @param model: PyTorch model whose parameters will be changed for every layer
    @param param_func: (nn.Parameter) -> nn.Parameter
                        Modifies parameter.
    @param module_func: (nn.Module) -> nn.Module
                        Optional function modifies module itself before param-level changes.
    @param param_check_func: (nn.Parameter) -> Bool
                            Optional function that determines if this param should be changed.
    @param memo_key_func: (nn.Parameter) -> Any
                        Optional function to use for memo keys. If None, uses param_key
    @param memo: Cache that prevents parameters from being operated on twice
    @return: Updated parameter model.
    """
    if not isinstance(model, torch.nn.Module):
        return model

    # Sets up module before param level changes. Params will be iterated from old model
    new_model = module_func(model) if module_func is not None else model

    # This is for any shared parameters for two different sections of the net
    memo = {} if memo is None else memo

    # 1) Re-write all parameters using _parameters field to preserve gradients
    model_params = getattr(new_model, '_parameters', [])
    for param_key in model_params:
        param = model._parameters[param_key]
        if param is not None and param_check_func(param):
            memo_key = memo_key_func(param) if memo_key_func is not None else param_key
            if memo_key in memo:
                new_model._parameters[param_key] = memo[memo_key]
            else:
                new_param = param_func(param)
                new_model._parameters[param_key] = new_param
                memo[memo_key] = new_param

    # TODO: Confirm if need buffers or not, coz I don't think so

    # 2) Recursively applyt to each sub-module
    submodules = getattr(new_model, '_modules', [])
    for submodule_key in submodules:
        submodule = submodules[submodule_key]
        new_model._modules[submodule_key] = apply_to_model_parameters(
            submodule,
            param_func,
            module_func=module_func,
            memo_key_func=memo_key_func,
            memo=memo
        )

    # Need to do this for RNNs apparently
    if hasattr(new_model, 'flatten_parameters'):
        new_model = new_model._apply(lambda x: x)

    return new_model


def compute_masked_loss(loss_fn, preds, targets, invalid_value=-1):
    """
    Computes mean loss between preds and targets, masking out the loss value for
    invalid entries in the 'targets' tensor (i.e. the loss is set to 0 for invalid entries)
    @param [nn.Module] loss_fn: PyTorch module computing loss with reduction='none'
    @param [torch.Tensor] preds: Model predictions in tensor form
    @param [torch.Tensor] targets: Ground truth targets in tensor form.
    @param [int or float] invalid_value: Value of invalid entries in targets tensor
    @return: Mean loss
    """
    valid_mask = targets != invalid_value
    targets[~valid_mask] = 0
    loss_t = loss_fn(preds, targets)

    # Only compute loss for valid pixels
    return (loss_t * valid_mask).mean()


def get_display_indices(batch_index, batch_size, num_display, len_loader, curr_len_display):
    """
    Checks if current batch has the next display item(s) given the current number
    of items queued for display.
    @param batch_index: Index of current batch
    @param batch_size: Size of each batch of items
    @param num_display: Total intended number of items to display
    @param len_loader: Number of batches in the data loader
    @param curr_len_display: Current number of items queued for display
    @return: Indices within current batch if to display, else empty list.
    """
    b = batch_size
    n = num_display
    total = len_loader * b  # total number of items (i.e. images)

    if n == 0: return []

    # total_items/num_display gives number of items to skip before next index
    target = (total // n) * curr_len_display if (total // n) > 0 else curr_len_display

    # If target item idx is within curr batch, return the idx mod curr batch
    indices = []
    while batch_index * b <= target < min((batch_index + 1) * b, len_loader * b):
        indices.append(target % (batch_index * b) if batch_index * b > 0 else target)

        target = curr_len_display + len(indices)
        target *= (total // n) if (total // n) > 0 else 1

    return indices
