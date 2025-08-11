import streamlit as st
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, Descriptors, Lipinski
import tempfile
import os
import py3Dmol
import torch

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
        st.code(Chem.MolToSmiles(mol))
        
        
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
            # Load the pre-trained solubility prediction model
            @st.cache_resource
            def load_solubility_pipeline():
                from model.train import smiles_to_fp, SolubilityNN
                model_path = os.path.join(os.path.dirname(__file__), "solubility_model.pth")
                model = SolubilityNN()
                model.load_state_dict(model_path)
                model.eval()
                return model

            solubility_model = load_solubility_pipeline()

            # Calculate descriptors for prediction (example: MolLogP, MolWt)
            logp = Descriptors.MolLogP(mol)
            molwt = Descriptors.MolWt(mol)
            num_rotatable = Lipinski.NumRotatableBonds(mol)
            num_hdonors = Lipinski.NumHDonors(mol)
            num_hacceptors = Lipinski.NumHAcceptors(mol)

            features = [[logp, molwt, num_rotatable, num_hdonors, num_hacceptors]]
            predicted_solubility = solubility_model.predict(features)[0]

            st.write(f"**Predicted Solubility:** {predicted_solubility:.4f}")

if __name__ == "__main__":
    main()
