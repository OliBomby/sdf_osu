import matplotlib.pyplot as plt
import numpy as np
from constants import image_shape, flat_num


def plot_signed_distance_field(distance_field, label=None):
    dist_image = distance_field.reshape(image_shape)

    if label is not None:
        one_hot = np.zeros(flat_num)
        one_hot[label] = 1
        dist_image += 2 * one_hot.reshape(image_shape)

    plot_heatmap(dist_image, "Signed Distance Field", "Distance / Object Radius")


def plot_prediction(prediction):
    pred_image = prediction.reshape(image_shape)
    plot_heatmap(pred_image, "Predicted Next Position", "Probability")


def plot_heatmap(img, title, y_label):
    # Create a figure and axes
    fig, ax = plt.subplots()

    # Plot the signed distance field
    im = ax.imshow(img, interpolation='nearest')

    # Add a colorbar for reference
    cbar = plt.colorbar(im)
    cbar.ax.set_ylabel(y_label)

    # Set the title and labels
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    # Show the plot
    plt.show()
