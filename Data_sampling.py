from dataset import UCSDAnomalyDataset
import matplotlib.pyplot as plt 

dataset = UCSDAnomalyDataset('./data/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Train')
sample_index = 0

sample_sequence = dataset[sample_index]

fig, axes = plt.subplots(1, len(sample_sequence), figsize=(15,3))
for i, image in enumerate(sample_sequence):
    image_np = image.permute(1, 2, 0).numpy()
    
    axes[i].imshow(image_np, cmap='gray')
    axes[i].axis('off')
    
plt.tight_layout()
plt.show()