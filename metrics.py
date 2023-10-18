from constants import coordinates_flat
import torch


def metric1(pred, ground_truth_indices, radius=3):
    """
    Calculate the sum of softmax predictions within a circle centered at the ground truth pixel.

    Parameters:
    pred (torch.Tensor): The predictions tensor (e.g., Nxflat_num).
    ground_truth_indices (torch.Tensor): The ground truth indices in flattened format (e.g., Nx1).
    coordinates_flat (torch.Tensor): The mapping from flat indices to 2D coordinates (e.g., flat_numx2).
    radius (int): The radius of the circle.

    Returns:
    torch.Tensor: A tensor containing the sum of softmax predictions within the circle for each image in the batch.
    """

    # Define the dimensions of the spatial layout
    H, W = 384, 512  # specific to your image dimensions

    # Number of samples in the batch
    N = pred.shape[0]

    # Reshape predictions to 2D spatial format
    pred_reshaped = pred.view(N, H, W)

    # Convert ground truth indices to 2D coordinates
    gt_coordinates = coordinates_flat[ground_truth_indices].squeeze(1)  # Nx2

    # Prepare to collect sums of softmax predictions within the circles
    sum_softmax_within_circle = torch.zeros(N, dtype=pred.dtype, device=pred.device)

    # Process each image in the batch
    for i in range(N):
        center_y, center_x = gt_coordinates[i]  # Extract the coordinates

        # Create a grid of (i,j) coordinates
        i_range = torch.arange(0, H, dtype=torch.float32, device=pred.device).view(1, H, 1).expand(1, H, W)
        j_range = torch.arange(0, W, dtype=torch.float32, device=pred.device).view(1, 1, W).expand(1, H, W)

        # Calculate the squared distance from each coordinate to the center
        dist = (i_range - center_y) ** 2 + (j_range - center_x) ** 2

        # Create a binary mask for the circle
        mask = (dist <= radius ** 2).float()

        # Apply softmax to the predictions
        softmax_pred = torch.softmax(pred_reshaped[i].view(-1), dim=0).view(H, W)

        # Apply the mask to the softmax predictions and sum up the values within the circle
        masked_softmax = softmax_pred * mask
        sum_softmax_within_circle[i] = torch.sum(masked_softmax)

    return sum_softmax_within_circle
