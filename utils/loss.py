import torch


def differentiable_histogram_batched(values, num_bins=244, value_range=(0, 1)):
    """
    values: (B, N)
    returns: (B, num_bins)
    """
    B, N = values.shape
    device = values.device
    bin_centers = torch.linspace(value_range[0], value_range[1], steps=num_bins, device=device)  # (num_bins,)
    bin_width = bin_centers[1] - bin_centers[0]  # scalar

    # Compute pairwise distances to bin centers → (B, N, num_bins)
    diffs = torch.abs(values.unsqueeze(-1) - bin_centers[None, None, :])  # (B, N, num_bins)

    # Triangular weighting
    weights = torch.clamp(1 - diffs / bin_width, min=0.0)  # (B, N, num_bins)

    # Sum over values to get histogram
    hist = weights.sum(dim=1)  # (B, num_bins)

    # Normalize histograms
    hist = hist / (hist.sum(dim=1, keepdim=True) + 1e-6)  # (B, num_bins)

    return hist  # (B, num_bins)

def histogram_loss_batched(pred, target, num_bins=244, value_range=(0, 1)):
    """
    pred, target: (B, N)
    returns: scalar loss
    """
    pred_hist = differentiable_histogram_batched(pred, num_bins=num_bins, value_range=value_range)   # (B, num_bins)
    target_hist = differentiable_histogram_batched(target, num_bins=num_bins, value_range=value_range)  # (B, num_bins)

    loss_per_sample = torch.sum((pred_hist - target_hist) ** 2, dim=1)  # (B,)
    return loss_per_sample.mean()  # scalar