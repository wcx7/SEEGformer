""" Utils for working with SSL models """

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved

import math
import warnings
from typing import Iterable, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from numpy.typing import NDArray
from torch.nn import Module
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.parameter import Parameter


def fft(signal, fs, fft_points = 4000, freq_used = 50):
    batch_Size = signal.shape[0]
    signal = np.asarray(signal)

    real_parts = np.zeros([batch_Size,signal.shape[1],int(fft_points/fs*freq_used)])
    imaginary_parts = np.zeros([batch_Size,signal.shape[1],int(fft_points/fs*freq_used)])
    amplitude = np.zeros([batch_Size,signal.shape[1],int(fft_points/fs*freq_used)])
    for i in range(batch_Size):
        # 执行FFT
        fft_result = np.fft.fft(signal[i,:,:], n = fft_points)
        # frequencies = np.fft.fftfreq(fft_points, 1/fs)  # 获取频率轴
            # 只保留正频率部分
        # positive_frequencies = frequencies[:fft_points//2]
        # reconstructed_signal = np.fft.ifftn(positive_fft_result[:,:int((fft_points/fs)*freq_used)],n = fft_points)
        positive_fft_result = fft_result[:,:fft_points//2]  # 只保留正频部分
        positive_fft_result = positive_fft_result[:,:int((fft_points/fs)*freq_used)]
        # 提取实数部分
        real_parts[i,:,:] = np.real(positive_fft_result)
        # 提取虚数部分
        imaginary_parts[i,:,:] = np.imag(positive_fft_result)
        # 计算幅值，取对数
        amplitude[i,:,:] = np.log(np.abs(positive_fft_result))

    return signal,real_parts,imaginary_parts,amplitude

def external_weights_get(final_position_encoding,batch_size,device):
    distance = np.zeros([final_position_encoding.shape[0],final_position_encoding.shape[0]])
    for i in range(final_position_encoding.shape[0]):
        for j in range(final_position_encoding.shape[0]):
            x = (final_position_encoding[i][0]-final_position_encoding[j][0])**2
            y = (final_position_encoding[i][1]-final_position_encoding[j][1])**2
            z = (final_position_encoding[i][2]-final_position_encoding[j][2])**2
            distance[i,j] = (x+y+z)**0.5      # 计算得到欧氏距离

        # 采用高斯核的方式    
    # std = np.std(distance)
    # distance = distance**2/(std**2)    
    # distance = np.exp(-distance)
        # min-max归一化再用1减去
    distance = (distance - np.min(distance))/(np.max(distance) - np.min(distance))
    distance = 1-distance

    external_weights = distance
    external_weights = np.tile(external_weights, (batch_size, 1, 1))
    external_weights = torch.Tensor(external_weights).to(device)

    return external_weights


@torch.no_grad()
def batch_shuffle(
    batch: torch.Tensor, distributed: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Randomly shuffles all tensors in the batch.

    Args:
        batch:
            The batch to shuffle.
        distributed:
            If True then batches are shuffled across multiple gpus.

    Returns:
        A (batch, shuffle) tuple where batch is the shuffled version of the
        input batch and shuffle is an index to restore the original order.

    Examples:
        >>> # forward pass through the momentum model with batch shuffling
        >>> x1_shuffled, shuffle = batch_shuffle(x1)
        >>> f1 = moco_momentum(x1)
        >>> out0 = projection_head_momentum(f0)
        >>> out1 = batch_unshuffle(out1, shuffle)
    """
    if distributed:
        return batch_shuffle_distributed(batch)
    batch_size = batch.shape[0]
    shuffle = torch.randperm(batch_size, device=batch.device)
    return batch[shuffle], shuffle


@torch.no_grad()
def batch_unshuffle(
    batch: torch.Tensor,
    shuffle: torch.Tensor,
    distributed: bool = False,
) -> torch.Tensor:
    """Unshuffles a batch.

    Args:
        batch:
            The batch to unshuffle.
        shuffle:
            Index to unshuffle the batch.
        distributed:
            If True then the batch is unshuffled across multiple gpus.

    Returns:
        The unshuffled batch.

    Examples:
        >>> # forward pass through the momentum model with batch shuffling
        >>> x1_shuffled, shuffle = batch_shuffle(x1)
        >>> f1 = moco_momentum(x1)
        >>> out0 = projection_head_momentum(f0)
        >>> out1 = batch_unshuffle(out1, shuffle)
    """
    if distributed:
        return batch_unshuffle_distributed(batch, shuffle)
    unshuffle = torch.argsort(shuffle)
    return batch[unshuffle]


@torch.no_grad()
def concat_all_gather(x: torch.Tensor) -> torch.Tensor:
    """Returns concatenated instances of x gathered from all gpus.

    This code was taken and adapted from here:
    https://github.com/facebookresearch/moco.

    """
    output = [torch.empty_like(x) for _ in range(dist.get_world_size())]
    dist.all_gather(output, x, async_op=False)
    output = torch.cat(output, dim=0)
    return output


@torch.no_grad()
def batch_shuffle_distributed(batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Shuffles batch over multiple devices.

    This code was taken and adapted from here:
    https://github.com/facebookresearch/moco.

    Args:
        batch:
            The tensor to shuffle.

    Returns:
        A (batch, shuffle) tuple where batch is the shuffled version of the
        input batch and shuffle is an index to restore the original order.

    """
    # gather from all devices
    batch_size_this = batch.shape[0]
    batch_gather = concat_all_gather(batch)
    batch_size_all = batch_gather.shape[0]

    num_devices = batch_size_all // batch_size_this

    # random shuffle index
    idx_shuffle = torch.randperm(batch_size_all, device=batch.device)

    # broadcast to all devices
    dist.broadcast(idx_shuffle, src=0)

    # index for restoring
    shuffle = torch.argsort(idx_shuffle)

    # shuffled index for this device
    rank = dist.get_rank()
    idx_this = idx_shuffle.view(num_devices, -1)[rank]
    return batch_gather[idx_this], shuffle


@torch.no_grad()
def batch_unshuffle_distributed(
    batch: torch.Tensor, shuffle: torch.Tensor
) -> torch.Tensor:
    """Undo batch shuffle over multiple devices.

    This code was taken and adapted from here:
    https://github.com/facebookresearch/moco.

    Args:
        batch:
            The tensor to unshuffle.
        shuffle:
            Index to restore the original tensor.

    Returns:
        The unshuffled tensor.

    """
    # gather from all devices
    batch_size_this = batch.shape[0]
    batch_gather = concat_all_gather(batch)
    batch_size_all = batch_gather.shape[0]

    num_devices = batch_size_all // batch_size_this

    # restored index for this gpu
    rank = dist.get_rank()
    idx_this = shuffle.view(num_devices, -1)[rank]
    return batch_gather[idx_this]


def deactivate_requires_grad(model: nn.Module):
    """Deactivates the requires_grad flag for all parameters of a model.

    This has the same effect as permanently executing the model within a `torch.no_grad()`
    context. Use this method to disable gradient computation and therefore
    training for a model.

    Examples:
        >>> backbone = resnet18()
        >>> deactivate_requires_grad(backbone)
    """
    for param in model.parameters():
        param.requires_grad = False


def activate_requires_grad(model: nn.Module):
    """Activates the requires_grad flag for all parameters of a model.

    Use this method to activate gradients for a model (e.g. after deactivating
    them using `deactivate_requires_grad(...)`).

    Examples:
        >>> backbone = resnet18()
        >>> activate_requires_grad(backbone)
    """
    for param in model.parameters():
        param.requires_grad = True


@torch.no_grad()
def update_momentum(model: nn.Module, model_ema: nn.Module, m: float):
    """Updates parameters of `model_ema` with Exponential Moving Average of `model`

    Momentum encoders are a crucial component fo models such as MoCo or BYOL.

    Examples:
        >>> backbone = resnet18()
        >>> projection_head = MoCoProjectionHead()
        >>> backbone_momentum = copy.deepcopy(moco)
        >>> projection_head_momentum = copy.deepcopy(projection_head)
        >>>
        >>> # update momentum
        >>> update_momentum(moco, moco_momentum, m=0.999)
        >>> update_momentum(projection_head, projection_head_momentum, m=0.999)
    """
    for model_ema, model in zip(model_ema.parameters(), model.parameters()):
        model_ema.data = model_ema.data * m + model.data * (1.0 - m)


@torch.no_grad()
def normalize_weight(weight: nn.Parameter, dim: int = 1, keepdim: bool = True):
    """Normalizes the weight to unit length along the specified dimension."""
    weight.div_(torch.norm(weight, dim=dim, keepdim=keepdim))


# copy paste from PyTorch master branch as it is not available in older releases
# source: https://github.com/pytorch/pytorch/blob/20ac7362009dd8e0aca6e72fc9357773136a83b8/torch/nn/init.py#L22-L54
def _no_grad_trunc_normal(
    tensor: torch.Tensor,
    mean: float,
    std: float,
    a: float,
    b: float,
) -> torch.Tensor:
    """Initializes the input tensor with a truncated normal distribution.

    This method is based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf

    Args:
        tensor:
            The tensor to initialize.
        mean:
            Mean of the distribution.
        std:
            Standard deviation of the distribution.
        a:
            Minimum value of the distribution, values below will be clamped.
        b:
            Maximum value of the distribution, values above will be clamped.

    """

    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2,
        )

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def repeat_token(token: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    """Repeats a token size times.

    Args:
        token:
            Token tensor with shape (1, 1, dim).
        size:
            (batch_size, sequence_length) tuple.

    Returns:
        Tensor with shape (batch_size, sequence_length, dim) containing copies
        of the input token.

    """
    batch_size, sequence_length = size
    return token.repeat(batch_size, sequence_length, 1)


def expand_index_like(index: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    """Expands the index along the last dimension of the input tokens.

    Args:
        index:
            Index tensor with shape (batch_size, idx_length) where each entry is
            an index in [0, sequence_length).
        tokens:
            Tokens tensor with shape (batch_size, sequence_length, dim).

    Returns:
        Index tensor with shape (batch_size, idx_length, dim) where the original
        indices are repeated dim times along the last dimension.

    """
    dim = tokens.shape[-1]
    index = index.unsqueeze(-1).expand(-1, -1, dim)
    return index


def get_at_index(tokens: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """Selects tokens at index.

    Args:
        tokens:
            Token tensor with shape (batch_size, sequence_length, dim).
        index:
            Index tensor with shape (batch_size, index_length) where each entry is
            an index in [0, sequence_length).

    Returns:
        Token tensor with shape (batch_size, index_length, dim) containing the
        selected tokens.

    """
    index = expand_index_like(index, tokens)
    return torch.gather(tokens, 1, index)


def set_at_index(
    tokens: torch.Tensor, index: torch.Tensor, value: torch.Tensor
) -> torch.Tensor:
    """Copies all values into the input tensor at the given indices.

    Args:
        tokens:
            Tokens tensor with shape (batch_size, sequence_length, dim).
        index:
            Index tensor with shape (batch_size, index_length).
        value:
            Value tensor with shape (batch_size, index_length, dim).

    Returns:
        Tokens tensor with shape (batch_size, sequence_length, dim) containing
        the new values.

    """
    index = expand_index_like(index, tokens)
    return torch.scatter(tokens, 1, index, value)


def mask_at_index(
    tokens: torch.Tensor, index: torch.Tensor, mask_token: torch.Tensor
) -> torch.Tensor:
    """Copies mask token into the input tensor at the given indices.

    Args:
        tokens:
            Tokens tensor with shape (batch_size, sequence_length, dim).
        index:
            Index tensor with shape (batch_size, index_length).
        mask_token:
            Value tensor with shape (1, 1, dim).

    Returns:
        Tokens tensor with shape (batch_size, sequence_length, dim) containing
        the new values.

    """
    mask = tokens.new_zeros(tokens.shape)
    mask = set_at_index(mask, index, 1)
    return (1 - mask) * tokens + mask * mask_token


def prepend_class_token(
    tokens: torch.Tensor, class_token: torch.Tensor
) -> torch.Tensor:
    """Prepends class token to tokens.

    Args:
        tokens:
            Tokens tensor with shape (batch_size, sequence_length, dim).
        class_token:
            Class token with shape (1, 1, dim).

    Returns:
        Tokens tensor with the class token prepended at index 0 in every
        sequence. The tensor has shape (batch_size, sequence_length + 1, dim).
    """
    batch_size = tokens.shape[0]
    batch_class_token = class_token.expand(batch_size, -1, -1)
    return torch.cat([batch_class_token, tokens], dim=1)


def patchify(images: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Converts a batch of input images into patches.

    Args:
        images:
            Images tensor with shape (batch_size, channels, height, width)
        patch_size:
            Patch size in pixels. Image width and height must be multiples of
            the patch size.

    Returns:
        Patches tensor with shape (batch_size, num_patches, channels * patch_size ** 2)
        where num_patches = image_width / patch_size * image_height / patch_size.

    """
    # N, C, H, W = (batch_size, channels, height, width)
    N, C, H, W = images.shape
    assert H == W and H % patch_size == 0

    patch_h = patch_w = H // patch_size
    num_patches = patch_h * patch_w
    patches = images.reshape(shape=(N, C, patch_h, patch_size, patch_w, patch_size))
    patches = torch.einsum("nchpwq->nhwpqc", patches)
    patches = patches.reshape(shape=(N, num_patches, patch_size**2 * C))
    return patches


def random_token_mask(
    size: Tuple[int, int],
    mask_ratio: float = 0.6,
    mask_class_token: bool = False,
    device: Optional[Union[torch.device, str]] = None,
) -> torch.Tensor:
    """Creates random token masks.

    Args:
        size:
            Size of the token batch for which to generate masks.
            Should be (batch_size, sequence_length).
        mask_ratio:
            Percentage of tokens to mask.
        mask_class_token:
            If False the class token is never masked. If True the class token
            might be masked.
        device:
            Device on which to create the index masks.

    Returns:
        A (index_keep, index_mask) tuple where each index is a tensor.
        index_keep contains the indices of the unmasked tokens and has shape
        (batch_size, num_keep). index_mask contains the indices of the masked
        tokens and has shape (batch_size, sequence_length - num_keep).
        num_keep is equal to sequence_length * (1- mask_ratio).

    """
    batch_size, sequence_length = size
    num_keep = int(sequence_length * (1 - mask_ratio))

    noise = torch.rand(batch_size, sequence_length, device=device)
    if not mask_class_token and sequence_length > 0:
        # make sure that class token is not masked
        noise[:, 0] = -1
        num_keep = max(1, num_keep)

    # get indices of tokens to keep
    indices = torch.argsort(noise, dim=1)
    idx_keep = indices[:, :num_keep]
    idx_mask = indices[:, num_keep:]

    return idx_keep, idx_mask


def nearest_neighbors(
    input_maps: torch.Tensor,
    candidate_maps: torch.Tensor,
    distances: torch.Tensor,
    num_matches: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Finds the nearest neighbors of the maps in input_maps in candidate_maps.

    Args:
        input_maps:
            A tensor of maps for which to find nearest neighbors.
            It has size: [batch_size, input_map_size, feature_dimension]
        candidate_maps:
            A tensor of maps to search for nearest neighbors.
            It has size: [batch_size, candidate_map_size, feature_dimension]
        distances:
            A tensor of distances between the maps in input_maps and candidate_maps.
            It has size: [batch_size, input_map_size, candidate_map_size]
        num_matches:
            Number of nearest neighbors to return. If num_matches is None or -1,
            all the maps in candidate_maps are considered.

    Returns:
        A tuple of tensors, containing the nearest neighbors in input_maps and candidate_maps.
        They both have size: [batch_size, input_map_size, feature_dimension]
    """

    if num_matches is None or num_matches == -1 or num_matches > input_maps.size(1):
        num_matches = input_maps.size(1)

    # Find nearest neighbour of each input element in the candidate map
    topk_values, topk_indices = distances.topk(
        k=1, dim=2, largest=False
    )  # [bsz, input_map_size, 1]
    topk_values = topk_values.squeeze(-1)  # [bsz, input_map_size]

    # Select num_matches neighbors pairs having the lowest distance value.
    _, min_indices = topk_values.topk(
        k=num_matches, dim=1, largest=False
    )  # [bsz, num_matches]

    # Create the filtered input map with num_matches lowest distance values.
    feature_dimension = input_maps.shape[2]
    filtered_input_maps = torch.gather(
        input_maps, 1, min_indices.unsqueeze(-1).expand(-1, -1, feature_dimension)
    )  # [bsz, num_matches, feature_dimension]

    # Create candidate maps in the same way as input maps, but using corrispondent candidate values
    selected_candidate_maps = torch.gather(
        candidate_maps, 1, topk_indices.expand(-1, -1, feature_dimension)
    )  # [bsz, input_map_size, feature_dimension]
    filtered_candidate_maps = torch.gather(
        selected_candidate_maps,
        1,
        min_indices.unsqueeze(-1).expand(-1, -1, feature_dimension),
    )  # [bsz, num_matches, feature_dimension]

    return filtered_input_maps, filtered_candidate_maps


def get_weight_decay_parameters(
    modules: Iterable[Module],
    decay_batch_norm: bool = False,
    decay_bias: bool = False,
) -> Tuple[List[Parameter], List[Parameter]]:
    """Returns all parameters of the modules that should be decayed and not decayed.

    Args:
        modules:
            List of modules to get the parameters from.
        no_batch_norm:
            If True, batch norm parameters are decayed.
        no_bias:
            If True, bias parameters are decayed.

    Returns:
        (params, params_no_weight_decay) tuple.
    """
    params = []
    params_no_weight_decay = []
    for module in modules:
        for mod in module.modules():
            if isinstance(mod, _BatchNorm):
                if not decay_batch_norm:
                    params_no_weight_decay.extend(mod.parameters(recurse=False))
                else:
                    params.extend(mod.parameters(recurse=False))
            else:
                for name, param in mod.named_parameters(recurse=False):
                    if not decay_bias and name.endswith("bias"):
                        params_no_weight_decay.append(param)
                    else:
                        params.append(param)
    return params, params_no_weight_decay


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    return _no_grad_trunc_normal(tensor, mean, std, a, b)


def apply_masks(x, masks):
    """
    :param x: tensor of shape [B (batch-size), N (num-patches), D (feature-dim)]
    :param masks: list of tensors containing indices of patches in [N] to keep
    """
    all_x = []
    for m in masks:
        mask_keep = m.unsqueeze(-1).repeat(1, 1, x.size(-1))
        all_x += [torch.gather(x, dim=1, index=mask_keep)]
    return torch.cat(all_x, dim=0)


def repeat_interleave_batch(x, B, repeat):
    N = len(x) // B
    x = torch.cat(
        [
            torch.cat([x[i * B : (i + 1) * B] for _ in range(repeat)], dim=0)
            for i in range(N)
        ],
        dim=0,
    )
    return x


def get_2d_sincos_pos_embed(
    embed_dim: int, grid_size: int, cls_token: bool = False
) -> NDArray[np.float_]:
    """Returns 2D sin-cos embeddings. Code from [0].

    - [0]: https://github.com/facebookresearch/ijepa

    Args:
        embed_dim:
            Embedding dimension.
        grid_size:
            Grid height and width. Should usually be set to sqrt(sequence length).
        cls_token:
            If True, a positional embedding for the class token is prepended to the returned
            embeddings.

    Returns:
        Positional embeddings array with size (grid_size * grid_size, embed_dim) if cls_token is False.
        If cls_token is True, a (1 + grid_size * grid_size, embed_dim) array is returned.
    """
    grid_h = np.arange(grid_size, dtype=float)
    grid_w = np.arange(grid_size, dtype=float)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(
    embed_dim: int, grid: NDArray[np.int_]
) -> NDArray[np.float_]:
    """Returns 2D sin-cos embeddings grid. Code from [0].

    - [0]: https://github.com/facebookresearch/ijepa

    Args:
        embed_dim:
            Embedding dimension.
        grid:
            2-dimensional grid to embed.

    Returns:
        Positional embeddings array with size (grid_size * grid_size, embed_dim).
    """
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed(
    embed_dim: int, grid_size: int, cls_token: bool = False
) -> NDArray[np.float_]:
    """Returns 1D sin-cos embeddings. Code from [0].

    - [0]: https://github.com/facebookresearch/ijepa

    Args:
        embed_dim:
            Embedding dimension.
        grid_size:
            Grid height and width. Should usually be set to sqrt(sequence length).
        cls_token:
            If True, a positional embedding for the class token is prepended to the returned
            embeddings.

    Returns:
        Positional embeddings array with size (grid_size, embed_dim) if cls_token is False.
        If cls_token is True, a (1 + grid_size, embed_dim) array is returned.
    """
    grid = np.arange(grid_size, dtype=float)
    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_1d_sincos_pos_embed_from_grid(
    embed_dim: int, pos: NDArray[np.int_]
) -> NDArray[np.float_]:
    """Returns 1D sin-cos embeddings grid. Code from [0].

    - [0]: https://github.com/facebookresearch/ijepa

    Args:
        embed_dim:
            Embedding dimension.
        pos:
            1-dimensional grid to embed.

    Returns:
        Positional embeddings array with size (grid_size, embed_dim).
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb
