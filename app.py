import streamlit as st
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, Descriptors, Lipinski
from rdkit.Chem import AllChem, DataStructs
import numpy as np
import tempfile
import os
import py3Dmol
import torch
import torch.nn as nn
import joblib

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

def smiles_to_fp(smiles, radius=2, nBits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
    # Fixed: Create array with correct size
    arr = np.zeros((nBits,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

def mol_to_3Dview(mol, size=(400, 400), style="stick", surface=False, opacity=0.5):
    """Draw molecule in 3D using py3Dmol."""
    AllChem.EmbedMolecule(mol)
    AllChem.MMFFOptimizeMolecule(mol)

    mb = Chem.MolToMolBlock(mol)
    viewer = py3Dmol.view(width=size[0], height=size[1])
    viewer.addModel(mb, 'mol')
    viewer.setStyle({'model': -1}, {style: {'color': 'spectrum'}})
    if surface:
        viewer.addSurface(py3Dmol.SAS, {'opacity': opacity})
    viewer.zoomTo()
    return viewer

def main():
    st.title("💊 Chemical Compound Visualizer")

    # Sidebar options
    st.sidebar.header("Options")
    input_type = st.sidebar.selectbox("Input Type", ["SMILES", "Molfile"])
    view_type = st.sidebar.selectbox("View Type", ["2D", "3D"])

    # Input area
    st.header("Input Compound")
    mol = None

    if input_type == "SMILES":
        smiles = st.text_input("Enter SMILES string:", "CCO")
        if smiles:
            mol = Chem.MolFromSmiles(smiles)
    else:
        molfile = st.file_uploader("Upload Molfile", type=["mol", "sdf"])
        if molfile:
            mol = Chem.MolFromMolBlock(molfile.getvalue().decode())

    if st.button("Visualize"):
        if mol is None:
            st.error("Invalid molecule. Please check your input.")
            return
        
        # Display SMILES
        st.subheader("Canonical SMILES")
        canonical_smiles = Chem.MolToSmiles(mol)
        st.code(canonical_smiles)
        
        col1, col2 = st.columns(2)
        with col1:
            # Visualization
            if view_type == "2D":
                img = Draw.MolToImage(mol, size=(400, 400))
                st.image(img, caption="2D Molecular Structure")
            else:
                st.write("3D Visualization:")
                viewer = mol_to_3Dview(mol)
                viewer_html = viewer._make_html()
                st.components.v1.html(viewer_html, height=450)
        
        with col2:
            # Molecular properties
            st.subheader("Molecular Properties")
            st.write(f"**Molecular Weight:** {Descriptors.ExactMolWt(mol):.2f}")
            st.write(f"**Number of Heavy Atoms:** {mol.GetNumHeavyAtoms()}")
            st.write(f"**Rotatable Bonds:** {Lipinski.NumRotatableBonds(mol)}")
            
            # Solubility prediction with proper error handling
            try:
                @st.cache_resource
                def load_pca():
                    pca_path = os.path.join(os.path.dirname(__file__), "pca.joblib")
                    if not os.path.exists(pca_path):
                        st.warning("PCA model file not found. Solubility prediction unavailable.")
                        return None
                    return joblib.load(pca_path)
                
                @st.cache_resource
                def load_solubility_model():
                    model_path = os.path.join(os.path.dirname(__file__), "solubility_model.pth")
                    if not os.path.exists(model_path):
                        st.warning("Solubility model file not found. Prediction unavailable.")
                        return None
                    model = SolubilityNN(128)  # Expecting 128 features after PCA
                    state_dict = torch.load(model_path, map_location="cpu")
                    model.load_state_dict(state_dict)
                    model.eval()
                    return model
                
                def predict_solubility(smiles_str):
                    pca = load_pca()
                    model = load_solubility_model()
                    
                    if pca is None or model is None:
                        return None
                    
                    # Generate fingerprint
                    fp = smiles_to_fp(smiles_str)
                    if fp is None:
                        return None
                    
                    # Apply PCA transformation (only once!)
                    X = fp.reshape(1, -1)
                    X_transformed = pca.transform(X)
                    
                    # Ensure we have the right number of features
                    if X_transformed.shape[1] != 128:
                        st.error(f"PCA output has {X_transformed.shape[1]} features, expected 128")
                        return None
                    
                    # Convert to tensor and predict
                    X_tensor = torch.FloatTensor(X_transformed)
                    with torch.no_grad():
                        prediction = model(X_tensor)
                    
                    return prediction.item()
                
                # Make prediction
                predicted_solubility = predict_solubility(canonical_smiles)
                
                if predicted_solubility is not None:
                    st.write(f"`Predicted Solubility:` {predicted_solubility:.4f}")
                else:
                    st.write("**Predicted Solubility:** Not available")
                    
            except Exception as e:
                st.error(f"Error in solubility prediction: {str(e)}")
                st.write("**Predicted Solubility:** Error occurred")

if __name__ == "__main__":
    main()