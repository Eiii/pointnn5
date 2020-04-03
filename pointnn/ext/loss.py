""" Pytorch 'extensions'. Just custom loss functions for now."""
import torch


def cham_dist(pred, target, reduction='mean'):
    # One-way distance function
    def _d(a, b):
        # Pair each point in a with each point in b
        # Unsqueeze+repeat over different dims so we get a cross product matrix
        repeat_a = a.unsqueeze(3)
        repeat_a = torch.cat([repeat_a]*b.shape[2], 3)
        repeat_b = b.unsqueeze(2)
        repeat_b = torch.cat([repeat_b]*a.shape[2], 2)
        # Subtract, square, sum to calculate square distance per pair
        sqr_diff = (repeat_a - repeat_b).pow(2).sum(dim=1)
        # Select closest (value w/ min square distance)
        min_diff, _ = sqr_diff.min(dim=2)
        sum_diff = min_diff.sum(dim=1)
        # Return total sum of closest distances across entire batch
        if reduction == 'mean':
            r = torch.mean(sum_diff)
        elif reduction == 'sum':
            r = torch.sum(sum_diff)
        elif reduction == 'none':
            r = sum_diff
        return r
    # Return sum of both sides of the distance function
    return _d(pred, target) + _d(target, pred)

def cham_dist_2(pred, target, reduction='mean'):
    # One-way distance function
    def _d(a, b):
        # Pair each point in a with each point in b
        # Unsqueeze+repeat over different dims so we get a cross product matrix
        repeat_a = a.unsqueeze(2)
        repeat_a = torch.cat([repeat_a]*b.shape[1], 2)
        repeat_b = b.unsqueeze(1)
        repeat_b = torch.cat([repeat_b]*a.shape[1], 1)
        # Subtract, square, sum to calculate square distance per pair
        sqr_diff = (repeat_a - repeat_b).pow(2).sum(dim=3)
        # Select closest (value w/ min square distance)
        min_diff, _ = sqr_diff.min(dim=2)
        min_diff = min_diff.pow(0.5)
        mean_diff = min_diff.mean(dim=1)
        mean_diff = mean_diff.pow(2)
        # Return total sum of closest distances across entire batch
        if reduction == 'mean':
            r = torch.mean(mean_diff)
        elif reduction == 'sum':
            r = torch.sum(mean_diff)
        elif reduction == 'none':
            r = mean_diff
        return r
    # Return sum of both sides of the distance function
    return _d(pred, target) + _d(target, pred)
