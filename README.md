# PyTorch Transfer Learning with Inception v3

This project demonstrates how to use transfer learning with PyTorch, specifically using the Inception v3 model as a feature extractor to classify images into two categories: cats and dogs. Transfer learning allows you to leverage pretrained models on large datasets and adapt them for your own tasks, significantly reducing training time and improving performance with smaller datasets.

## Project Structure

- **Load Data**: Load and transform images for training and validation using `torchvision.datasets.ImageFolder`.
- **General Functions**: Utility functions to train the model and visualize predictions.
- **Transfer Learning (Inception v3)**: Configure the Inception v3 model for transfer learning, freezing all layers except the final classification layer.
- **Train and Evaluate**: Train the model, fine-tuning only the final layers, and evaluate its performance.

## Setup Instructions

### 1. Load Data
We use the `ImageFolder` dataset format, where images are organized into subdirectories by class (e.g., `dogs/` and `cats/`).

Images are transformed using `torchvision.transforms`:
- Resize to 325x325 pixels.
- CenterCrop to 299x299 pixels (required for Inception v3).
- Normalize to the standard mean and standard deviation for pretrained models.

You can configure the data directory to point to the `dogs-vs-cats` dataset used in this project.

```python
data_dir = "../input/dogs-vs-cats/dataset/dataset"
```

The dataset structure should be:
```
data_dir/
    training_set/
        cats/
        dogs/
    test_set/
        cats/
        dogs/
```

### 2. General Functions for Training and Visualization
Functions are provided to:
- Train the model with learning rate scheduling and checkpointing of the best model.
- Visualize predictions after training using matplotlib.

```python
def train_model(model, criterion, optimizer, scheduler, num_epochs=2, is_inception=False):
    # Training loop implementation
```

### 3. Transfer Learning: Inception v3
We use the Inception v3 model, pre-trained on ImageNet, as a fixed feature extractor. The model's layers are frozen, except for the final classification layer, which is updated to classify between 2 categories: cats and dogs.

```python
model_ft = models.inception_v3(pretrained=True)
for param in model_ft.parameters():
    param.requires_grad = False

# Replace final layer
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 2)
```

### 4. Train and Evaluate
The model is trained using stochastic gradient descent (SGD) with a learning rate scheduler. The learning rate decays by a factor of 0.1 every epoch.

```python
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=1, gamma=0.1)
model_ft = train_model(model_ft, criterion, optimizer_conv, exp_lr_scheduler, num_epochs=2, is_inception=True)
```

### Visualizing Model Predictions
After training, predictions on the validation set are visualized to assess the model’s performance.

```python
visualize_model(model_ft)
```

### Dependencies

- Python 3.6+
- PyTorch
- Torchvision
- Matplotlib
- Numpy

### How to Run

1. Clone the repository and navigate to the project directory.
2. Install the necessary dependencies:
   ```
   pip install torch torchvision matplotlib numpy
   ```
3. Prepare the dataset in the `data_dir` as mentioned above.
4. Run the main script to train and evaluate the model:
   ```
   python main.py
   ```
5. Visualize the results using the provided visualization functions.

### Notes

- The model will use GPU acceleration if available (`torch.cuda.is_available()`).
- The script saves and reloads the best-performing model based on validation accuracy during training.

### Acknowledgments

This project references pretrained models from PyTorch's torchvision library and dataset formats commonly used for image classification tasks.
