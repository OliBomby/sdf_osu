from constants import coordinates_flat, flat_num
import torch
import numpy as np
from scipy.ndimage import distance_transform_edt


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


def histogram_plot(pred, input, distance_step, cicle_radius, max_distance=None):
    """
    Create a histogram representing the summed values of predictions at various distances
    from the nearest non-zero pixel in the input image.

    Parameters:
    pred (torch.Tensor): The tensor with predicted probabilities for each position (Batch x Height x Width).
    input (torch.Tensor): The input tensor with the same shape as pred, where non-zero values indicate points of interest.
    distance_step (float): The step size for the distance bins in the histogram.
    max_distance (float, optional): The maximum distance to consider. If None, it will be calculated.

    Returns:
    np.ndarray: A histogram representing the summed probabilities at various distances.
    """

    # Batch size
    batch_size = pred.shape[0]

    # Ensure the input tensor is binary (i.e., consists of 0s and 1s)
    input_binary = (input != 0).int()

    # Convert PyTorch tensors to numpy arrays for scipy compatibility
    pred_np = pred.cpu().numpy()
    input_binary_np = input_binary.cpu().numpy()

    # Placeholder for the distance transforms
    distances = np.zeros_like(input_binary_np, dtype=np.float32)

    # Compute the distance transform for each item in the batch separately
    for i in range(batch_size):
        distances[i] = distance_transform_edt(1 - input_binary_np[i])

    # If maximum distance is not provided, calculate it from the distances
    if max_distance is None:
        max_distance = np.max(distances)

    # Define the bins based on the maximum distance and distance step
    bins = np.arange(0, max_distance + distance_step, distance_step)
    # Normalize by dividing with circle radius, and then multiplying by 4 to get pixel values on the real map
    bins_normalized = bins / cicle_radius * 4

    # We will calculate the histogram per batch item but sum the results, so the histogram size remains constant
    histogram = np.zeros(len(bins) - 1, dtype=np.float32)

    # Calculate the histograms in a batched manner for efficiency
    for i in range(batch_size):
        # Calculate the distances for the current batch item
        current_distances = distances[i].reshape(-1)
        current_pred = pred_np[i].reshape(-1)

        # Calculate the histogram for the current item in the batch
        hist, _ = np.histogram(current_distances, bins=bins_normalized, weights=current_pred)

        # Add the current histogram to the accumulated histogram
        histogram += hist

    return histogram
