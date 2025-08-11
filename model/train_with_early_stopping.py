import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import DataLoader
from utils.early_stopping import EarlyStopping

# Example GCN model class
class GCNModel(nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes):
        super(GCNModel, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, edge_index, edge_attr):
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        x = self.dropout(x)
        x = self.conv3(x, edge_index, edge_attr)
        return x

def train_model(train_loader, val_loader, model, optimizer, criterion, device, epochs=100):
    early_stopping = EarlyStopping(patience=10, verbose=True)
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.edge_attr)
            loss = criterion(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                out = model(data.x, data.edge_index, data.edge_attr)
                loss = criterion(out[data.val_mask], data.y[data.val_mask])
                val_loss += loss.item()
        
        # Calculate average losses
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        print(f'Epoch {epoch+1}/{epochs}')
        print(f'Train Loss: {train_loss:.4f}')
        print(f'Val Loss: {val_loss:.4f}')
        
        # Early stopping
        early_stopping(val_loss, model)
        
        if early_stopping.early_stop:
            print("Early stopping")
            break

    # Load the best model
    model.load_state_dict(torch.load('checkpoint.pt'))
    return model

# Example usage
if __name__ == '__main__':
    # Example parameters (you'll need to adjust these based on your actual data)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_features = 20  # Adjust based on your node feature size
    hidden_channels = 64
    num_classes = 1  # For regression
    
    # Create model
    model = GCNModel(num_features, hidden_channels, num_classes)
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Example data loaders (you'll need to replace these with your actual data)
    # train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Train the model
    # model = train_model(train_loader, val_loader, model, optimizer, criterion, device)
