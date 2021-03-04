import os
import torch
import torch.nn as nn
from typing import Callable, Any, Optional

from segmentation import MODELS, LOADERS
from data_loaders.dataset import CropDataset


def create_dirs(*dirs):
    """
    Creates directories based on paths passed in as arguments.
    """

    def f_mkdir(p):
        if not os.path.isdir(p):
            print(f"Creating directory {p}")
            os.makedirs(p)

    for p in dirs:
        f_mkdir(p)


def create_model(model_config, num_classes):
    """
    Creates a new segmentation model given the config dictionary.
    Uses the specialised creator functions for each model.
    """
    assert model_config["classifier"].lower() in MODELS, \
        "Please specify a valid segmenation classifier available in MODELS"

    # Create Segmentation model
    seg_model_class = MODELS[model_config["classifier"].lower()]
    seg_model = seg_model_class.create(model_config, num_classes)

    return seg_model


def load_model(model_config, num_classes, from_checkpoint=None, freeze_backbone=False, new_head=False):
    """
    Loads a segmentation model based on its config dictionary.
    If specified, load's model weights from a checkpoint file.
    Else, creates a new, fresh instance of the model.
    If specified, also freezes all parameters in backbone layer.
    Can have relaxed loading with new_head=True to allow
    for loading feature extractor w/ random init head
    """
    model = create_model(model_config, num_classes)

    if from_checkpoint:
        if type(from_checkpoint) is str:
            checkpoint_path = from_checkpoint
            assert os.path.isfile(from_checkpoint), \
                f"Model's .bin checkpoint file doesn't exist at: {checkpoint_path}"
        else:
            raise ValueError(f"Keyword arg `from_checkpoint` must be a string")

        print(f"Loading model weights from checkpoint path {checkpoint_path}")

        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else "cpu")

        state_dict = torch.load(checkpoint_path, map_location=device)
        if new_head:
            if model_config['name'] == 'simple_net':
                # Phrasing as below allows for mjutaton in iteration
                for k in list(state_dict.keys()):
                    if 'final_conv' in k:
                        del state_dict[k]
            else:
                raise NotImplementedError
            model.load_state_dict(state_dict, strict=False)
        else:
            model.load_state_dict(state_dict)

    # TODO: Finetuning (freezing layers other than backbone)
    # Freeze backbone if specified
    if freeze_backbone:
        print("Freezing backbone layers...")
        if hasattr(model, "backbone"):
            for param in model.backbone.parameters():
                param.requires_grad = False
        elif hasattr(model, "conv_layers"):
            for param in model.conv_layers.parameters():
                param.requires_grad = False
        else:
            raise NotImplementedError

    return model


def save_model(model, save_path):
    """
    Saves a segmentation model to `save_path`.
    If using multiple gpus/data parallelism, then save `.module` attribute.
    """
    if torch.cuda.device_count() > 1:
        torch.save(model.module.state_dict(), save_path)
    else:
        torch.save(model.state_dict(), save_path)
    print(f"Saved model weights at: {save_path}")


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


def create_dataset(classifier_name, *args, **kwargs) -> CropDataset:
    """
    Creates a new initialised CropDataset based on the type of classifier.
    """
    assert classifier_name in LOADERS, \
        "Please specify a valid segmenation classifier available in MODELS"

    dataset_type = LOADERS[classifier_name]
    return dataset_type(*args, **kwargs)
