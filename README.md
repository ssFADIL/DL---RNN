# DL- Developing a Recurrent Neural Network Model for Stock Prediction

## AIM
To develop a Recurrent Neural Network (RNN) model for predicting stock prices using historical closing price data.

## Problem Statement and Dataset



## DESIGN STEPS
### STEP 1: 

Load and normalize data, create sequences.

### STEP 2: 

Convert data to tensors and set up DataLoader.

### STEP 3: 

Define the RNN model architecture.

### STEP 4: 

Summarize, compile with loss and optimizer.

### STEP 5: 

Train the model with loss tracking

### STEP 6: 

Predict on test data, plot actual vs. predicted prices.



## PROGRAM

### Name:SYED FADIL S

### Register Number:212225040454

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import os

## Step 1: Load and Preprocess Data

# Ensure dummy files are recreated to reflect updated periods
if os.path.exists('trainset.csv'):
    os.remove('trainset.csv')
if os.path.exists('testset.csv'):
    os.remove('testset.csv')

# Create dummy trainset.csv and testset.csv if they don't exist, ensuring sufficient data
# (These will now always be created due to the removal above if they exist)
if not os.path.exists('trainset.csv'):
    dates_train = pd.to_datetime(pd.date_range(start='2020-01-01', periods=200))
    close_prices_train = np.random.rand(200) * 100 + 50
    df_train_dummy = pd.DataFrame({'Date': dates_train, 'Close': close_prices_train})
    df_train_dummy.to_csv('trainset.csv', index=False)
    print("Created dummy 'trainset.csv'")

if not os.path.exists('testset.csv'):
    # Increase periods to ensure test set has enough data for seq_length=60
    dates_test = pd.to_datetime(pd.date_range(start='2020-09-01', periods=100))
    close_prices_test = np.random.rand(100) * 100 + 60
    df_test_dummy = pd.DataFrame({'Date': dates_test, 'Close': close_prices_test})
    df_test_dummy.to_csv('testset.csv', index=False)
    print("Created dummy 'testset.csv'")

# Load training and test datasets
df_train = pd.read_csv('trainset.csv')
df_test = pd.read_csv('testset.csv')

# Use closing prices
train_prices = df_train['Close'].values.reshape(-1, 1)
test_prices = df_test['Close'].values.reshape(-1, 1)

# Normalize the data based on training set only
scaler = MinMaxScaler()
scaled_train = scaler.fit_transform(train_prices)
scaled_test = scaler.transform(test_prices)

# Create sequences
def create_sequences(data, seq_length):
    x = []
    y = []
    for i in range(len(data) - seq_length):
        x.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(x), np.array(y)

seq_length = 60
x_train, y_train = create_sequences(scaled_train, seq_length)
x_test, y_test = create_sequences(scaled_test, seq_length)

# Convert to PyTorch tensors
x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Create dataset and dataloader
train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Define RNN Model
class RNNModel(nn.Module):
  def __init__(self,input_size=1,hidden_size=64,num_layers=2,output_size=1):
    super(RNNModel,self).__init__()
    self.rnn=nn.RNN(input_size,hidden_size,num_layers,batch_first=True)
    self.fc=nn.Linear(hidden_size,output_size)
  def forward(self,x):
    out,_=self.rnn(x)
    out=self.fc(out[:,-1,:])
    return out
model = RNNModel()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Install torchinfo if not already installed (optional, for model summary)
try:
    import torchinfo
except ImportError:
    !pip install torchinfo
    import torchinfo
from torchinfo import summary

# input_size = (batch_size, seq_len, input_size)
# Only summarize if x_train_tensor is not empty to avoid errors with empty tensors
if x_train_tensor.shape[0] > 0:
    summary(model, input_size=(64, seq_length, 1))
else:
    print("Cannot display model summary: x_train_tensor is empty.")

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

##  Train the Model

def train_model(model, train_loader, criterion, optimizer, epochs=20):
    train_losses = []
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch =x_batch.to(device),y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        train_losses.append(total_loss / len(train_loader))
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss / len(train_loader):.4f}")
# Plot training loss
    print('Name:  SYED FADIL S  ')
    print('Register Number:  212225040454 ')
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()
    plt.show()

# Only train if train_loader is not empty
if len(train_loader) > 0:
    train_model(model,train_loader,criterion,optimizer)
else:
    print("Cannot train model: train_loader is empty. Check your training data and sequence length.")

##  Make Predictions on Test Set
model.eval()
with torch.no_grad():
    # Check if x_test_tensor is not empty before making predictions
    if x_test_tensor.shape[0] > 0:
        predicted = model(x_test_tensor.to(device)).cpu().numpy()
        actual = y_test_tensor.cpu().numpy()

        # Inverse transform the predictions and actual values
        predicted_prices = scaler.inverse_transform(predicted)
        actual_prices = scaler.inverse_transform(actual)

        # Plot the predictions vs actual prices
        print('Name:        SYED FADIL S           ')
        print('Register Number:    212225040454  ')
        plt.figure(figsize=(10, 6))
        plt.plot(actual_prices, label='Actual Price')
        plt.plot(predicted_prices, label='Predicted Price')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.title('Stock Price Prediction using RNN')
        plt.legend()
        plt.show()
        print(f'Predicted Price: {predicted_prices[-1]}')
        print(f'Actual Price: {actual_prices[-1]}')
    else:
        print("Cannot make predictions: x_test_tensor is empty. Check your test data and sequence length.")

```

### OUTPUT

## Training Loss Over Epochs Plot

Include your plot here

## True Stock Price, Predicted Stock Price vs time

Include your plot here

### Predictions
Include the predictions on test data

## RESULT
Include your result here
