import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import time

# Function to precompute B-spline basis functions
def precompute_bspline_basis(num_splines, degree, num_points=100):
    knots = np.linspace(0, 1, num_splines + degree + 1)
    x = np.linspace(0, 1, num_points)
    basis = np.zeros((num_points, num_splines))

    def cox_de_boor(x, k, d, knots):
        if d == 0:
            return np.where((knots[k] <= x) & (x < knots[k+1]), 1.0, 0.0)
        else:
            a = (x - knots[k]) / (knots[k+d] - knots[k] + 1e-8)
            b = (knots[k+d+1] - x) / (knots[k+d+1] - knots[k+1] + 1e-8)
            return a * cox_de_boor(x, k, d-1, knots) + b * cox_de_boor(x, k+1, d-1, knots)

    for i in range(num_splines):
        basis[:, i] = cox_de_boor(x, i, degree, knots)

    return torch.tensor(basis, dtype=torch.float32)

# Precompute basis functions
num_splines = 10
degree = 3
basis = precompute_bspline_basis(num_splines, degree).to('cuda')  # Move the precomputed basis to GPU

# Define the PrecomputedB_Spline class
class PrecomputedB_Spline(nn.Module):
    def __init__(self, precomputed_basis):
        super(PrecomputedB_Spline, self).__init__()
        self.precomputed_basis = precomputed_basis
        self.coefficients = nn.Parameter(torch.randn(precomputed_basis.size(1)) * 0.1).to('cuda')  # Initialize coefficients
        self.w = nn.Parameter(torch.ones(1).to('cuda'))
    
    def forward(self, x):
        idx = (x * (self.precomputed_basis.size(0) - 1)).long()
        idx = torch.clamp(idx, 0, self.precomputed_basis.size(0) - 1)
        basis = self.precomputed_basis[idx]
        spline = torch.matmul(basis, self.coefficients)
        b = x / (1 + torch.exp(-x))  # Silu function
        return self.w * (b + spline)

# Define the KANLayer class
class KANLayer(nn.Module):
    def __init__(self, in_features, out_features, precomputed_basis):
        super(KANLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.b_splines = nn.ModuleList(
            [PrecomputedB_Spline(precomputed_basis) for _ in range(out_features)]
        )
        self.weights = nn.Parameter(torch.randn(out_features, in_features) * 0.1).to('cuda')  # Initialize weights
        self.bias = nn.Parameter(torch.zeros(out_features)).to('cuda')
        self.batch_norm = nn.BatchNorm1d(out_features)  # Batch normalization

    def forward(self, x):
        batch_size = x.size(0)
        activation_output = []
        for i in range(self.out_features):
            activation = self.b_splines[i]
            linear_combination = torch.matmul(x, self.weights[i]) + self.bias[i]
            activation_output.append(activation(linear_combination).unsqueeze(1))
        output = torch.cat(activation_output, dim=1)
        return self.batch_norm(output)

# Define the KANModel class
class KANModel(nn.Module):
    def __init__(self, precomputed_basis):
        super(KANModel, self).__init__()
        self.layer1 = KANLayer(784, 1024, precomputed_basis)  # Increased number of neurons
        self.layer2 = KANLayer(1024, 512, precomputed_basis)
        self.layer3 = KANLayer(512, 256, precomputed_basis)
        self.layer4 = KANLayer(256, 10, precomputed_basis)  # Added an additional layer

    def forward(self, x):
        x = x.view(-1, 784)  # Flatten the input image
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

# Load the MNIST dataset
print("Loading MNIST dataset...")
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=256, shuffle=True)  # Adjusted batch size to 4
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=256, shuffle=False)
print("MNIST dataset loaded.")

# Define the model, loss function, and optimizer
print("Initializing model...")
model = KANModel(basis).to('cuda')  # Move the model to GPU
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0167, betas=(0.577, .839))  # Adjusted learning rates to 0.0167 for cotangent inverse of precomputed basis number of splines and degrees, for the betas the cotangent lower beta is cotangent of 30 degrees and the upper beta is cotangent of 40 degrees. lower beta cotangent is number of splines * number of degrees and upper beta is lower beat plus degree number of splines which is 10 more in this usecase equaling 40cotangent. 
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)  # Learning rate scheduler
print("Model initialized.")

# Train the model
print("Starting training...")
start_time = time.time()
for epoch in range(1):  # Increased number of epochs for example
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to('cuda'), labels.to('cuda')

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 100 == 99:    # Print every 100 mini-batches
            print(f'[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 100:.3f}')
            running_loss = 0.0

#    scheduler.step()  # Step the learning rate scheduler
    print(f'Finished epoch {epoch + 1} loss: {running_loss / 100:.3f}')
end_time = time.time()
training_time = end_time - start_time
print(f'Total training time: {training_time:.2f} seconds')

print('Finished Training')

# Evaluate the model
print("Evaluating model...")
start_eval_time = time.time()
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to('cuda'), labels.to('cuda')
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

end_eval_time = time.time()
evaluation_time = end_eval_time - start_eval_time
print(f'Accuracy of the network on the 10000 test images: {100 * correct / total}%')
print(f'Total evaluation time: {evaluation_time:.2f} seconds')
