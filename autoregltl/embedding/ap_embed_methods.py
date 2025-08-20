import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from autoregltl.positional_encoding import positional_encoding

func_list = {
    "randn": torch.randn,
}
register_func = lambda f: func_list.setdefault(f.__name__, f)


@register_func
def pos(ap_count, d_ap, **factory_kwargs):
    m = positional_encoding(ap_count, d_ap, **factory_kwargs).view(ap_count, d_ap)
    return m[torch.randperm(ap_count), :]


@register_func
def nbor_naive(num_points, num_dimensions, **factory_kwargs):
    # Generate all possible neighbor offsets in n-dimensional space: {-1, 0, 1}^num_dimensions
    choices = torch.tensor([-1, 0, 1], **factory_kwargs)
    offsets = torch.cartesian_prod(*[choices for _ in range(num_dimensions)])
    
    # Remove the central point (all zeros)
    offsets = [v for v in offsets if torch.any(v != 0)]
    
    # Randomly sample the requested number of points from the set
    sampled_neighbors = random.sample(offsets, num_points)
    
    # Convert the sampled neighbors back into a tensor
    return torch.stack(sampled_neighbors)


@register_func
def diagbor_naive(num_points, num_dimensions, **factory_kwargs):
    # Generate all possible neighbor offsets in n-dimensional space: {-1, 1}^num_dimensions
    choices = torch.tensor([-1, 1], **factory_kwargs)
    offsets = torch.cartesian_prod(*[choices for _ in range(num_dimensions)])

    offsets = list(offsets)
    
    # Randomly sample the requested number of points from the set
    sampled_neighbors = random.sample(offsets, num_points)
    
    # Convert the sampled neighbors back into a tensor
    return torch.stack(sampled_neighbors)


@register_func
def diagbor(num_points, num_dimensions, **factory_kwargs):
    # Generate all possible neighbor offsets in n-dimensional space: {-1, 1}^num_dimensions
    # First, generate num_points numbers between 0 and 2^num_dimensions
    # Then, convert each number to its binary representation
    # Finally, convert each binary representation to a tensor of {-1, 1}
    # This will give us a tensor of size (num_points, num_dimensions)
    return torch.tensor([
        [-1 if b == '1' else 1 for b in bin(i)[2:].zfill(num_dimensions)]
        for i in random.sample(range(2**num_dimensions), num_points)
    ], **factory_kwargs)


@register_func
def nbor(num_points, num_dimensions, **factory_kwargs):
    # Generate all possible neighbor offsets in n-dimensional space: {-1, 0, 1}^num_dimensions
    # First, generate num_points numbers between 0 and 3^num_dimensions
    # Then, convert each number to its ternary representation
    # Finally, convert each ternary representation to a tensor of {-1, 0, 1}
    # Unlike diagbor, we need to remove the central point (all zeros)
    mid_point = (3**num_points - 1) // 2
    numbers = [
        i + 1 if i >= mid_point else i
        for i in random.sample(range(3**num_dimensions-1), num_points)
    ]
    return torch.tensor([
        [int(b)-1 for b in np.base_repr(i, base=3).zfill(num_dimensions)]
        for i in numbers
    ], **factory_kwargs)


# diagbor without uniqueness checks
@register_func
def diagbor_no_check(num_points, num_dimensions, **factory_kwargs):
    return (torch.randint(0, 2, (num_points, num_dimensions), **factory_kwargs) * 2) - 1


# nbor without uniqueness checks
@register_func
def nbor_no_check(num_points, num_dimensions, **factory_kwargs):
    out = torch.randint(-1, 2, (num_points, num_dimensions), **factory_kwargs)
    # Check if there is a row with all zeros
    while torch.any(torch.all(out == 0, dim=1)):
        # If there is, regenerate the row
        out[torch.all(out == 0, dim=1)] = torch.randint(-1, 2, (torch.sum(torch.all(out == 0, dim=1)), num_dimensions), **factory_kwargs)
    return out


def get_ap_method(name):
    return func_list[name]


class LearnableShuffledEmbeds(nn.Module):
    def __init__(
            self,
            token_count,
            d_ap,
            **factory_kwargs,
    ):
        self.token_count = token_count
        super().__init__()
        self.embeds = nn.parameter.Parameter(torch.empty((token_count, d_ap), **factory_kwargs))

    def reset_parameters(self) -> None:
        torch.nn.init.normal_(self.embeds)
    
    def forward(self, count, num_dimensions, **factory_kwargs):
        perm = torch.randperm(self.token_count, device=self.embeds.device)[:count]
        return self.embeds[perm]