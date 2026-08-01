"""Penalty functions for SAE activations."""

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.autograd as autograd


class VanillaPenalty(nn.Module):
    """No penalty - just pass through activations."""
    
    def forward(self, activations):
        return activations


class TopKPenalty(nn.Module):
    """TopK penalty - keep only top k activations per sample."""
    
    def __init__(self, top_k):
        super().__init__()
        self.top_k = top_k
    
    def forward(self, activations):
        acts_topk = torch.topk(activations, self.top_k, dim=-1)
        acts_topk = torch.zeros_like(activations).scatter(
            -1, acts_topk.indices, acts_topk.values
        )
        return acts_topk


class BatchTopKPenalty(nn.Module):
    """
    Batch Top-k penalty (Bussmann et al., 2024).

    During training, finds the k-th highest activation across the flattened
    batch and uses it as a threshold.  A running average of this threshold is
    maintained via exponential momentum so that at eval time the threshold is
    deterministic and batch-size independent.
    """

    def __init__(self, top_k, threshold_momentum=0.9):
        super().__init__()
        self.top_k = top_k
        self.threshold_momentum = threshold_momentum

    def forward(self, activations, threshold=None):
        mask = (activations >= threshold).float().detach()
        return activations * mask
    
    def _get_threshold(self, activations):
        acts_topk = torch.topk(activations.flatten(), self.top_k * activations.shape[0], dim = -1)
        return acts_topk.values[-1]



class MatryoshkaBatchTopKPenalty(BatchTopKPenalty):
    """
    Batch Top-k penalty with Matryoshka group masking.

    After applying the global BatchTopK threshold, features beyond the
    currently active groups are zeroed out.  ``active_groups`` defaults to
    all groups and can be reduced at inference time to use fewer features.
    """

    def __init__(self, top_k, group_sizes, threshold_momentum=0.9):
        super().__init__(top_k, threshold_momentum)
        self.group_sizes = group_sizes
        self.group_indices = [0] + torch.cumsum(torch.tensor(group_sizes), dim=0).tolist()
        self.active_groups = len(group_sizes)

    def forward(self, activations, threshold=None):
        acts = super().forward(activations, threshold)
        max_idx = self.group_indices[self.active_groups]
        if max_idx < acts.shape[-1]:
            acts = acts.clone()
            acts[..., max_idx:] = 0
        return acts


class L1Penalty(nn.Module):
    """L1 penalty - ReLU activation."""
    
    def forward(self, activations):
        return F.relu(activations)



class RectangleFunction(autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return ((x > -0.5) & (x < 0.5)).float()

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[(x <= -0.5) | (x >= 0.5)] = 0
        return grad_input


class JumpReLUFunction(autograd.Function):
    @staticmethod
    def forward(ctx, x, log_threshold, bandwidth):
        ctx.save_for_backward(x, log_threshold, torch.tensor(bandwidth))
        threshold = torch.exp(log_threshold)
        return x * (x > threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        x, log_threshold, bandwidth_tensor = ctx.saved_tensors
        bandwidth = bandwidth_tensor.item()
        threshold = torch.exp(log_threshold)
        x_grad = (x > threshold).float() * grad_output
        threshold_grad = (
            -(threshold / bandwidth)
            * RectangleFunction.apply((x - threshold) / bandwidth)
            * grad_output
        )
        return x_grad, threshold_grad, None


class JumpReLU(nn.Module):
    def __init__(self, feature_size, bandwidth, device='cpu'):
        super(JumpReLU, self).__init__()
        self.log_threshold = nn.Parameter(torch.zeros(feature_size, device=device))
        self.bandwidth = bandwidth

    def forward(self, x):
        return JumpReLUFunction.apply(x, self.log_threshold, self.bandwidth)


class StepFunction(autograd.Function):
    @staticmethod
    def forward(ctx, x, log_threshold, bandwidth):
        ctx.save_for_backward(x, log_threshold, torch.tensor(bandwidth))
        threshold = torch.exp(log_threshold)
        return (x > threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        x, log_threshold, bandwidth_tensor = ctx.saved_tensors
        bandwidth = bandwidth_tensor.item()
        threshold = torch.exp(log_threshold)
        x_grad = torch.zeros_like(x)
        threshold_grad = (
            -(1.0 / bandwidth)
            * RectangleFunction.apply((x - threshold) / bandwidth)
            * grad_output
        )
        return x_grad, threshold_grad, None
