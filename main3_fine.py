import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from models import get_model3  # Assuming you have defined your model in a separate 'models' module
from constants import image_shape
from data_loading_img import list_beatmap_files_from_ds, process_path3, list_beatmap_files_from_ds_with_sr
from plotting import plot_signed_distance_field, plot_prediction

# Create training dataset
batch_size = 32
transform = transforms.Compose([transforms.ToTensor()])
labeled_ds = list_beatmap_files_from_ds_with_sr(5, 15).skip(0).interleave(process_path3, cycle_length=1).shuffle(1000).batch(batch_size, drop_last=True)

# Build model
model = get_model3(image_shape, 4)
print(model)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

checkpoint_filepath_old = "saved_models/sdf_osu_5_fine.pth"
checkpoint_filepath = "saved_models/sdf_osu_5_fine_2.pth"

# Load pretrained weights if available
if torch.cuda.is_available():
    checkpoint = torch.load(checkpoint_filepath_old)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# Train the model
epochs = 1000
# for epoch in range(epochs):
#     for inputs, labels in labeled_ds:
#         optimizer.zero_grad()
#         outputs = model(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#     print(f"Epoch [{epoch + 1}/{epochs}] Loss: {loss.item()}")

# Save the trained model
# torch.save({
#     'epoch': epochs,
#     'model_state_dict': model.state_dict(),
#     'optimizer_state_dict': optimizer.state_dict(),
#     'loss': loss,
# }, checkpoint_filepath)

# Generate predictions for a subset of the validation set
num_val = 5
val_ds = list(list_beatmap_files_from_ds_with_sr(5, 15).skip(19).interleave(process_path3, cycle_length=num_val).skip(100).batch(num_val).take(1))[0]
val_preds = []
with torch.no_grad():
    for img in val_ds[0]:
        outputs = model(img.unsqueeze(0))
        val_preds.append(outputs.cpu().numpy())

for j in range(num_val):
    for i in range(3, -1, -1):
        plot_signed_distance_field(val_ds[0][j, i, :, :].numpy(), val_ds[1][j])
    plot_prediction(np.sqrt(val_preds[j]))
