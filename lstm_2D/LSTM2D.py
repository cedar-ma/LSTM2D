import torch
import torch.nn as nn
import lightning as pl
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR

class ConvLSTM2DCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.padding = kernel_size // 2  # Ensure spatial dimensions are preserved

        # Convolutional gates
        self.conv = nn.Conv2d(
            in_channels=input_channels + hidden_channels,
            out_channels=4 * hidden_channels,  # 4 gates: input, forget, cell, output
            kernel_size=kernel_size,
            padding=self.padding,
        )

    def forward(self, x, h, c):
        # x: input tensor of shape (batch_size, input_channels, height, width)
        # h: hidden state of shape (batch_size, hidden_channels, height, width)
        # c: cell state of shape (batch_size, hidden_channels, height, width)
        combined = torch.cat([x, h], dim=1)  # Concatenate input and hidden state
        gates = self.conv(combined)  # Apply convolution
        # Split the gates into input, forget, cell, and output
        input_gate, forget_gate, cell_gate, output_gate = gates.chunk(4, dim=1)
        # Apply activation functions
        input_gate = torch.sigmoid(input_gate)
        forget_gate = torch.sigmoid(forget_gate)
        cell_gate = torch.tanh(cell_gate)
        output_gate = torch.sigmoid(output_gate)
        # Update cell state and hidden state
        c = forget_gate * c + input_gate * cell_gate
        h = output_gate * torch.tanh(c)
        return h, c

class ConvLSTM2D(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, num_layers=1):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        self.cells = nn.ModuleList([
            ConvLSTM2DCell(input_channels if i == 0 else hidden_channels, hidden_channels, kernel_size)
            for i in range(num_layers)
        ])

    def forward(self, x, hidden_states=None):
        # x: input tensor of shape (batch_size, seq_len, input_channels, height, width)
        batch_size, seq_len, _, height, width = x.size()
        # Initialize hidden and cell states if not provided
        if hidden_states is None:
            h = [torch.zeros(batch_size, self.hidden_channels, height, width, device=x.device) for _ in range(self.num_layers)]
            c = [torch.zeros(batch_size, self.hidden_channels, height, width, device=x.device) for _ in range(self.num_layers)]
        else:
            h, c = hidden_states
        # Process each time step
        outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :, :, :]  # Current time step
            for layer in range(self.num_layers):
                h[layer], c[layer] = self.cells[layer](x_t, h[layer], c[layer])
                x_t = h[layer]  # Use the hidden state as input to the next layer
            outputs.append(h[-1].unsqueeze(1))  # Store the output of the last layer
        # Stack outputs into a tensor
        outputs = torch.cat(outputs, dim=1)  # shape: (batch_size, seq_len, hidden_channels, height, width)
        return outputs, (h, c)


class ConvLSTMModel(pl.LightningModule):
    def __init__(self, input_channels, hidden_channels, kernel_size, lr, num_layers=1):
        super().__init__()
        self.conv_lstm = ConvLSTM2D(input_channels, hidden_channels, kernel_size, num_layers)
        self.conv_out = nn.Conv2d(hidden_channels, input_channels, kernel_size, padding="same")
        self.criterion = nn.MSELoss()  # Loss function for regression
        self.lr = lr

    def forward(self, x, pred_len=30):
        # x shape: (batch_size, seq_len, channels, height, width)
        batch_size = x.size(0)
        # Initialize the hidden state for iterative prediction
        h, c = None, None
        # List to store predictions
        predictions = []
        # Iteratively predict the next `pred_len` time steps
        for _ in range(pred_len):
            # Forward pass through ConvLSTM
            conv_lstm_out, (h, c) = self.conv_lstm(x)  # conv_lstm_out shape: (batch_size, seq_len, hidden_channels, height, width)
            # Take the output of the last time step
            last_time_step_out = conv_lstm_out[:, -1, :, :, :]  # shape: (batch_size, hidden_channels, height, width)
            # Predict the next time step
            next_pred = self.conv_out(last_time_step_out)  # shape: (batch_size, channels, height, width)
            predictions.append(next_pred.unsqueeze(1))  # Add time step dimension
            # Update the input sequence: remove the first time step and append the prediction
            x = torch.cat([x[:, 1:, :, :, :], next_pred.unsqueeze(1)], dim=1)  # shape: (batch_size, seq_len, channels, height, width)
        # Stack predictions into a tensor
        predictions = torch.cat(predictions, dim=1)  # shape: (batch_size, pred_len, channels, height, width)
        return predictions

    def training_step(self, batch, batch_idx):
        x, y, _, _ = batch  # x: (batch_size, 10, channels, height, width), y: (batch_size, 30, channels, height, width)
        y = y.permute(0,3,1,2)
        y = y.unsqueeze(2)
        x = x.permute(0,3,1,2)
        x = x.unsqueeze(2)
        pred_size = y.size(1)
        # Predict the next 30 time steps
        predictions = self(x, pred_len=pred_size)  # shape: (batch_size, 30, channels, height, width)
        # Compute loss
        loss = self.criterion(predictions, y)
        self.log("loss", loss, on_step=False, on_epoch=True, logger=True, sync_dist=True)#rank_zero_only=True)
        current_lr = self.optimizers().param_groups[0]['lr']
        self.log('learning_rate', current_lr, on_step=False, on_epoch=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y, _, _ = batch
        y = y.permute(0,3,1,2)
        y = y.unsqueeze(2)
        x = x.permute(0,3,1,2)
        x = x.unsqueeze(2)
        pred_size = y.size(1)
        predictions = self(x, pred_len=pred_size)  # shape: (batch_size, 30, channels, height, width)
        # Compute loss
        val_loss = self.criterion(predictions, y)
        self.log("val_loss", val_loss, on_step=False, on_epoch=True, logger=True, sync_dist=True)#rank_zero_only=True)

        return val_loss
    
    def predict_step(self, batch, batch_idx):
        x, y, mins, maxs = batch
        y = y.permute(0,3,1,2)
        y = y.unsqueeze(2)
        x = x.permute(0,3,1,2)
        x = x.unsqueeze(2)
        pred_size = y.size(1)
        predictions = self(x, pred_len=pred_size)  # shape: (batch_size, 30, channels, height, width)
        predictions = predictions.permute(0, 2, 3, 4, 1)
        y = y.permute(0, 2, 3, 4, 1)

        yhat = self.back01_transform(predictions, mins, maxs)
        y = self.back01_transform(y, mins, maxs)
        print('Back transform prediction: ',yhat.min(),yhat.max())
        print('Back transform truth: ', y.min(), y.max())

        predictions = (yhat <= 0).float()
        y = (y <= 0).float()
        pred = {
            'j': y,
            'jhat': predictions,
        }

        return pred

    def back01_transform(self, normalized, original_mins, original_maxs):
        back_transformed = torch.zeros_like(normalized)
        original_mins = original_mins.flatten()
        original_maxs = original_maxs.flatten()
        for i in range(normalized.shape[-1]):
            #print(type(original_mins[i]))
            #print(original_mins[i])
            if original_maxs[i] != original_mins[i]:
                back_transformed[...,i] = normalized[...,i]*(original_maxs[i] - original_mins[i]) + original_mins[i]
            else:
                back_transformed[...,i] = normalized[...,i]

        return back_transformed

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=1)
        return {
        'optimizer': optimizer,
        'lr_scheduler': {
            'scheduler': scheduler,
            'monitor': 'val_loss',  # Metric to monitor
            "interval": "epoch",  # Check every epoch
            "frequency": 5
            }
         }



