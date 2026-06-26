import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class IoTNSMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(IoTNSMModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

def generate_dummy_data(num_samples, input_size, num_classes):
    X = np.random.rand(num_samples, input_size).astype(np.float32)
    y = np.random.randint(0, num_classes, num_samples)
    y_one_hot = np.zeros((num_samples, num_classes))
    y_one_hot[np.arange(num_samples), y] = 1
    return torch.tensor(X), torch.tensor(y_one_hot, dtype=torch.float32)

def train_model(model, criterion, optimizer, X_train, y_train, epochs=100):
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

def evaluate_model(model, X_test, y_test):
    with torch.no_grad():
        outputs = model(X_test)
        predicted = torch.argmax(outputs, dim=1)
        actual = torch.argmax(y_test, dim=1)
        accuracy = (predicted == actual).sum().item() / len(actual)
        print(f'Accuracy: {accuracy * 100:.2f}%')

if __name__ == '__main__':
    # Hyperparameters
    input_size = 10  # Number of features
    hidden_size = 16  # Number of hidden neurons
    output_size = 3  # Number of output classes (e.g., normal, anomaly, attack)
    learning_rate = 0.01
    epochs = 50
    num_samples = 1000

    # Generate dummy data
    X, y = generate_dummy_data(num_samples, input_size, output_size)

    # Split data into training and testing sets
    train_size = int(0.8 * num_samples)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Initialize model, loss function, and optimizer
    model = IoTNSMModel(input_size, hidden_size, output_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    train_model(model, criterion, optimizer, X_train, y_train, epochs)

    # Evaluate the model
    evaluate_model(model, X_test, y_test)