import torch
import torch.nn as nn
import torch.nn.functional as F
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
        
        input_gate = torch.sigmoid(input_gate)
        forget_gate = torch.sigmoid(forget_gate)
        cell_gate = torch.tanh(cell_gate)
        output_gate = torch.sigmoid(output_gate)
        
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
        
        if hidden_states is None:
            h = [torch.zeros(batch_size, self.hidden_channels, height, width, device=x.device) for _ in range(self.num_layers)]
            c = [torch.zeros(batch_size, self.hidden_channels, height, width, device=x.device) for _ in range(self.num_layers)]
        else:
            h, c = hidden_states
        
        outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :, :, :]  # Current time step
            for layer in range(self.num_layers):
                h[layer], c[layer] = self.cells[layer](x_t, h[layer], c[layer])
                x_t = h[layer]  # Use the hidden state as input to the next layer
            outputs.append(h[-1].unsqueeze(1))  # Store the output of the last layer
        
        outputs = torch.cat(outputs, dim=1)  # shape: (batch_size, seq_len, hidden_channels, height, width)
        return outputs, (h, c)


class SegmentedLoss(nn.Module):
    def __init__(self, weights=None):
        super(SegmentedLoss, self).__init__()
        # Define weights for each segment (default: equal weights)
        self.weights = weights if weights is not None else [1.0, 1.0, 1.0]

    def forward(self, y_pred, y_true):
        
        assert y_pred.shape == y_true.shape, "Shapes of predictions and targets must match"

        
        pred_1_10 = y_pred[:, :10]  # Steps 1-10
        true_1_10 = y_true[:, :10]

        pred_11_20 = y_pred[:, 10:20]  # Steps 11-20
        true_11_20 = y_true[:, 10:20]

        pred_21_30 = y_pred[:, 20:30]  # Steps 21-30
        true_21_30 = y_true[:, 20:30]

        
        loss_1_10 = F.mse_loss(pred_1_10, true_1_10)
        loss_11_20 = F.mse_loss(pred_11_20, true_11_20)
        loss_21_30 = F.mse_loss(pred_21_30, true_21_30)

        
        total_loss = (
            self.weights[0] * loss_1_10 +
            self.weights[1] * loss_11_20 +
            self.weights[2] * loss_21_30
        ) / sum(self.weights)

        return total_loss

class ConvLSTMModel(pl.LightningModule):
    def __init__(self, input_channels, hidden_channels, kernel_size, lr, num_layers=1):
        super().__init__()
        self.conv_lstm = ConvLSTM2D(input_channels, hidden_channels, kernel_size, num_layers)
        self.conv_out = nn.Conv2d(hidden_channels, input_channels, kernel_size, padding="same")
        self.criterion = SegmentedLoss(weights=[0.2, 0.3, 0.5])#nn.MSELoss()  # Segment Loss function
        self.lr = lr

    def forward(self, x, pred_len=30):
        # x shape: (batch_size, seq_len, channels, height, width)
        batch_size = x.size(0)
        # Initialize the hidden state for iterative prediction
        h, c = None, None
        
        predictions = []
        
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
        x, y, means, stds = batch  # x: (batch_size, height, width, 10), y: (batch_size, height, width, 30)
        y = y.permute(0,3,1,2) # x: (batch_size, 10, height, width)
        y = y.unsqueeze(2) # x: (batch_size, 10, channels, height, width)
        x = x.permute(0,3,1,2)
        x = x.unsqueeze(2)
        pred_size = y.size(1)
        
        predictions = self(x, pred_len=pred_size)  # shape: (batch_size, 30, channels, height, width)
        
        loss = self.criterion(predictions, y)
        self.log("loss", loss, on_step=False, on_epoch=True, logger=True, sync_dist=True)#rank_zero_only=True)
        current_lr = self.optimizers().param_groups[0]['lr']
        self.log('learning_rate', current_lr, on_step=False, on_epoch=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y, means, stds = batch
        y = y.permute(0,3,1,2)
        y = y.unsqueeze(2)
        x = x.permute(0,3,1,2)
        x = x.unsqueeze(2)
        pred_size = y.size(1)
        predictions = self(x, pred_len=pred_size)  # shape: (batch_size, 30, channels, height, width)
        
        val_loss = self.criterion(predictions, y)
        self.log("val_loss", val_loss, on_step=False, on_epoch=True, logger=True, sync_dist=True)#rank_zero_only=True)

        return val_loss
    
    def predict_step(self, batch, batch_idx):
        x, y, means, stds = batch
        y = y.permute(0,3,1,2)
        y = y.unsqueeze(2)
        x = x.permute(0,3,1,2)
        x = x.unsqueeze(2)
        pred_size = y.size(1)
        predictions = self(x, pred_len=pred_size)  # shape: (batch_size, 30, channels, height, width)
        predictions = predictions.permute(0, 2, 3, 4, 1) # shape: (batch_size, 1, height, width, 30)
        y = y.permute(0, 2, 3, 4, 1)

        print('Before transform prediction: ',predictions.min(),predictions.max())
        print('Before transform truth: ', y.min(), y.max())

        yhat = self.z_score_back_transform(predictions, means, stds)
        y = self.z_score_back_transform(y, means, stds)
        yhat[y==0] = 0 #mask solid phase
        print('Back transform prediction: ',yhat.min(),yhat.max())
        print('Back transform truth: ', y.min(), y.max())

        #predictions = (yhat <= 0).float()
        #y = (y <= 0).float()
        yhat[yhat>0] = 1
        yhat[yhat==0] = 2
        yhat[yhat<0] = 0
        y[y>0] = 1
        y[y==0] = 2
        y[y<0] = 0
        pred = {
            'j': y,
            'jhat': yhat,
        }

        return pred

    def z_score_back_transform(self, normalized_data, means, stds):
        """
        Back-transform Z-score normalized data to its original scale.
        Args:
            normalized_data (torch.Tensor): Normalized data of shape [batch_size, 1, height, width, seq_len].
            means (np.ndarray): Means of shape [batch_size, height, width, 1].
            stds (np.ndarray): Standard deviations of shape [batch_sizse, height, width, 1].
        Returns:
            original_data (torch.Tensor): Back-transformed data of shape [batch_size, 1, height, width, seq_len].
        """

        # Reshape means and stds to match the batch and channel dimensions
        means = means.unsqueeze(1)  # Shape: [1, 1, height, width, 1]
        stds = stds.unsqueeze(1)    # Shape: [1, 1, height, width, 1]

        
        original_data = (normalized_data * stds) + means
        return original_data

    def back_transform(self, normalized, original_mins, original_maxs):
        """
        back transform from range [-1,1]
        """
        back_transformed = torch.zeros_like(normalized)
        original_mins = original_mins.flatten()
        original_maxs = original_maxs.flatten()
        for i in range(normalized.shape[-1]):
            #print(type(original_mins[i]))
            #print(original_mins[i])
            if original_maxs[i] != original_mins[i]:
                back_transformed[...,i] = (normalized[...,i]+1)*(original_maxs[i] - original_mins[i])/2 + original_mins[i]
            else:
                back_transformed[...,i] = normalized[...,i]

        return back_transformed


    def back01_transform(self, normalized, original_mins, original_maxs):
        """
        back transform from range [0,1]
        """
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



