---
title: "Neural Network Knowledge Distillation - Code"
---

# Multi-Model Training with Different Subsets

In deep learning, training multiple specialized models on different subsets of data can significantly enhance efficiency and accuracy. This technique is particularly useful in scenarios where different data segments exhibit unique characteristics that a single model might struggle to capture effectively. In this blog, we will explore how to implement multi-model training using PyTorch and efficiently infer results from an input image.

## Why Train Multiple Specialist Models?

Training multiple specialist models instead of a single generalist model has several benefits:

- **Improved Accuracy:** Each model specializes in a particular subset of data, leading to better performance on specific tasks.
- **Efficient Inference:** Deploying smaller, specialized models allows for faster inference and reduced computational load.
- **Scalability:** Different models can be trained independently and combined when necessary, making the system more modular and scalable.

## Step 1: Preparing Data Subsets

To train specialist models, we first need to divide the dataset into different subsets based on specific criteria. In this example, we split the CIFAR-10 dataset into two subsets: one containing the first 5 classes and another containing the last 5 classes. Each subset will be used to train a separate model.

```python
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import torch
import torch.nn as nn
import torch.optim as optim

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Load full dataset
full_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

# Define subsets (e.g., splitting dataset into two groups: first 5 classes and last 5 classes)
subset_indices_1 = [i for i in range(len(full_dataset)) if full_dataset.targets[i] < 5]
subset_indices_2 = [i for i in range(len(full_dataset)) if full_dataset.targets[i] >= 5]

subset1 = Subset(full_dataset, subset_indices_1)
subset2 = Subset(full_dataset, subset_indices_2)

# Create data loaders
train_loader_1 = DataLoader(subset1, batch_size=64, shuffle=True)
train_loader_2 = DataLoader(subset2, batch_size=64, shuffle=True)
```

## Step 2: Defining and Training Specialist Models

Each subset of data will be used to train a separate instance of a simple Convolutional Neural Network (CNN). This allows each model to specialize in its respective subset, improving classification performance.

We begin by defining a basic CNN architecture. This model consists of a convolutional layer, an activation function, a pooling layer, and a fully connected layer to classify images.

```python
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 16 * 16, num_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x
```

Next, we create two instances of `SimpleCNN`: one for the first subset and another for the second subset. Each model is configured to classify five classes, corresponding to the dataset split.

```python
# Train two specialist models
model1 = SimpleCNN(num_classes=5)  # For first 5 classes
model2 = SimpleCNN(num_classes=5)  # For last 5 classes

# Define optimizer and loss function
criterion = nn.CrossEntropyLoss()
optimizer1 = optim.Adam(model1.parameters(), lr=0.001)
optimizer2 = optim.Adam(model2.parameters(), lr=0.001)
```

Now, we implement a training function that iterates through the dataset for multiple epochs. During each epoch, the model processes batches of images, computes loss, and updates weights using backpropagation.

```python
# Training loop (simplified for brevity)
def train_model(model, train_loader, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

train_model(model1, train_loader_1, optimizer1)
train_model(model2, train_loader_2, optimizer2)
```

## Step 3: Performing Inference with a Single Input Image

Once both models are trained, we need a mechanism to decide which model should be used for inference. One way to accomplish this is by comparing the confidence scores of each model and selecting the one with the highest probability.

We define an inference function that takes an input image and passes it through both models. The function then selects the model that produces the highest confidence score.

```python
import torch.nn.functional as F

def infer_image(image, model1, model2):
    """ Perform inference using the appropriate model. """
    model1.eval()
    model2.eval()

    with torch.no_grad():
        output1 = model1(image)
        output2 = model2(image)

    prob1 = F.softmax(output1, dim=1)
    prob2 = F.softmax(output2, dim=1)

    # Decide based on max probability
    if torch.max(prob1) > torch.max(prob2):
        return torch.argmax(prob1).item(), "Model 1"
    else:
        return torch.argmax(prob2).item() + 5, "Model 2"
```

Now, we simulate an input image and use our inference function to predict its class and determine which model was used.

```python
# Example image tensor (simulated for demo)
image = torch.randn(1, 3, 32, 32)  # Random CIFAR-10 sized image
predicted_class, model_used = infer_image(image, model1, model2)
print(f"Predicted Class: {predicted_class}, Model Used: {model_used}")
```

## Conclusion

Multi-model training with different subsets provides an effective way to specialize models for specific categories of data. This approach can lead to improved accuracy and efficiency, particularly in real-world applications with diverse datasets. By training specialized models and intelligently routing inference requests, we can achieve optimized performance while reducing computational complexity.
