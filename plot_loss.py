import os
import pandas as pd
import matplotlib.pyplot as plt

log_files = [
    ("lightning_logs\\version_6\\metrics.csv", 0),
]

for path, x_offset in log_files:
    # Read the CSV file into a DataFrame
    df = pd.read_csv(path)

    # Extract the columns
    steps = df['step']
    loss = df['loss_step']

    # Create a line plot for the loss
    plt.plot(steps, loss, label=os.path.basename(os.path.dirname(path)))

plt.title('Loss over steps')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.grid(True)
plt.yscale("log")
# plt.ylim(0, 0.08)
plt.legend()
plt.show()
