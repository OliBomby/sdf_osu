from constants import coordinates_flat, flat_num
import torch


def circle_accuracy(pred, ground_truth_indices, radius=30):
    """
    Calculate the sum of softmax predictions within a circle centered at the ground truth pixel.

    Parameters:
    pred (torch.Tensor): The tensor with predicted probabilities for each position.
    ground_truth_indices (torch.Tensor): The ground truth indices in flattened format.
    radius (float): The radius of the circle.

    Returns:
    torch.Tensor: A tensor containing the probability for predictions within radius pixels of the label.
    """
    coordinates_flat_good = coordinates_flat.to(pred.device)
    batch_size = pred.shape[0]
    batch_coordinates_flat = coordinates_flat_good.unsqueeze(0).expand(batch_size, -1, -1)  # (batch_size, flat_num, 2)
    gt_coordinates = coordinates_flat_good[ground_truth_indices]  # (batch_size, 2)
    gt_coordinates_2 = gt_coordinates.unsqueeze(1).expand(-1, flat_num, -1)  # (batch_size, flat_num, 2)
    distances = torch.sum(torch.square(gt_coordinates_2 - batch_coordinates_flat), dim=-1)  # (batch_size, flat_num)
    mask = distances <= radius ** 2  # (batch_size, flat_num)
    masked_pred = mask * pred
    total_probability = torch.sum(masked_pred, dim=1)

    return total_probability.mean()
