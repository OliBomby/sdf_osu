import os
import pandas as pd
import matplotlib.pyplot as plt

log_files = [
    # ("lightning_logs\\version_6\\metrics.csv", 0),
    ("lightning_logs\\version_7\\metrics.csv", 0),
]

for path, x_offset in log_files:
    # Read the CSV file into a DataFrame
    df = pd.read_csv(path)

    # Extract the columns
    filtered_df = df.dropna(subset=['train_loss_step'])
    train_steps = filtered_df['step']
    train_loss = filtered_df['train_loss_step']

    filtered_df = df.dropna(subset=['valid_loss_epoch'])
    valid_steps = filtered_df['step']
    valid_loss = filtered_df['valid_loss_epoch']

    # Create a line plot for the loss
    plt.plot(train_steps.values, train_loss.values, label=os.path.basename(os.path.dirname(path)) + " train")
    plt.plot(valid_steps.values, valid_loss.values, label=os.path.basename(os.path.dirname(path)) + " val")

plt.title('Loss over steps')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.grid(True)
plt.yscale("log")
# plt.ylim(0, 0.08)
plt.legend()
plt.show()
