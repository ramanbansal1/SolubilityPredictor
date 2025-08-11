import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
from sklearn.decomposition import PCA
import warnings
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import joblib
import sys

warnings.filterwarnings('ignore')

# === 1. Load Data ===
url = "https://github.com/dataprofessor/data/raw/master/delaney.csv"
df = pd.read_csv(url)
y = df["measured log(solubility:mol/L)"].values

# === 2. Convert SMILES to Morgan fingerprints ===
def smiles_to_fp(smiles, radius=2, nBits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
    arr = np.zeros((1,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

X = [smiles_to_fp(s) for s in df["SMILES"]]
mask = [x is not None for x in X]
X = np.array([x for x in X if x is not None])
y = y[mask]

# PCA reduction
pca = PCA(n_components=128)
X = pca.fit_transform(X)

# Save PCA for deployment
joblib.dump(pca, "pca.joblib")
print("PCA saved to pca.joblib")

import sys
sys.exit(0)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# === 3. Dataset & DataLoader ===
class MolDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = MolDataset(X_train, y_train)
test_dataset = MolDataset(X_test, y_test)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

# === 4. Neural network model ===
class SolubilityNN(nn.Module):
    def __init__(self, input_dim):
        super(SolubilityNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)

model = SolubilityNN(X.shape[1])
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

# === 5. Training loop with LR scheduler ===
losses = []
for epoch in range(500):
    model.train()
    running_loss = 0.0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        preds = model(batch_X)
        loss = criterion(preds, batch_y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        losses.append(loss.item())

    avg_train_loss = running_loss / len(train_loader)
    scheduler.step(avg_train_loss)  # adjust LR based on training loss

    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}, Loss: {avg_train_loss:.4f}")

plt.plot(losses)
plt.xlabel("Batch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.savefig("losses.png", bbox_inches='tight')

# === 6. Evaluation ===
model.eval()
with torch.no_grad():
    y_pred_list, y_true_list = [], []
    for batch_X, batch_y in test_loader:
        preds = model(batch_X)
        y_pred_list.append(preds.numpy())
        y_true_list.append(batch_y.numpy())

    y_pred = np.vstack(y_pred_list)
    y_true = np.vstack(y_true_list)

r2 = r2_score(y_true, y_pred)
rmse = mean_squared_error(y_true, y_pred)

print(f"R²: {r2:.3f}")
print(f"RMSE: {rmse:.3f}")


# === 7. Save the trained model ===
torch.save(model.state_dict(), "solubility_model.pth")
print("Model saved to solubility_model.pth")
