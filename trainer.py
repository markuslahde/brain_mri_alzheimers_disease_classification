import torch
import torch.nn as nn
from torch import optim


def train_model(model, train_loader, args):
    # 1. Set the model to training mode
    model.train() 

    # 2. Define the loss function
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 3. Iterate over the epochs
    for epoch in range(args.epochs):
        running_loss = 0.0

        for batch in train_loader:
            # Get the inputs and targets
            inputs = batch['img']
            targets = batch['target']

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)

            # Calculate loss
            loss = criterion(outputs, targets)

            # Optimization
            loss.backward()
            optimizer.step() # Update the model weights

            running_loss += loss.item()
        print(f'Epoch {epoch}, Loss: ', running_loss/len(train_loader))


def validate_model(model, val_loader, args):
    running_loss = 0.0
    
    criterion = nn.CrossEntropyLoss()

    # Set model to evaluation mode
    model.eval()

    for batch in val_loader:
        inputs = batch['img']
        targets = batch['target']

        outputs = model(inputs)

        # Calculate loss
        loss = criterion(outputs, targets)

        running_loss += loss.item()

        # Calculate accuracy
        # Accuracy function

    print('Validation Loss: ', running_loss/len(val_loader))




