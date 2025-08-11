import torch
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolDescriptors
import numpy as np

def get_atom_features(atom):
    """
    Get atom features for each atom in a molecule.
    Returns a feature vector of length 20.
    """
    features = []
    # Atom type
    atom_type = atom.GetSymbol()
    features.extend([1 if atom_type == 'C' else 0,
                    1 if atom_type == 'N' else 0,
                    1 if atom_type == 'O' else 0,
                    1 if atom_type == 'S' else 0,
                    1 if atom_type == 'F' else 0,
                    1 if atom_type == 'Cl' else 0,
                    1 if atom_type == 'Br' else 0,
                    1 if atom_type == 'I' else 0])
    
    # Atomic number
    features.append(atom.GetAtomicNum())
    
    # Degree
    features.append(atom.GetDegree())
    
    # Formal charge
    features.append(atom.GetFormalCharge())
    
    # Number of Hs
    features.append(atom.GetTotalNumHs())
    
    # Aromaticity
    features.append(1 if atom.GetIsAromatic() else 0)
    
    return np.array(features)

def get_bond_features(bond):
    """
    Get bond features for each bond in a molecule.
    Returns a feature vector of length 5.
    """
    features = []
    
    # Bond type
    bond_type = bond.GetBondType()
    features.extend([
        1 if bond_type == Chem.rdchem.BondType.SINGLE else 0,
        1 if bond_type == Chem.rdchem.BondType.DOUBLE else 0,
        1 if bond_type == Chem.rdchem.BondType.TRIPLE else 0,
        1 if bond_type == Chem.rdchem.BondType.AROMATIC else 0
    ])
    
    # Conjugation
    features.append(1 if bond.GetIsConjugated() else 0)
    
    return np.array(features)

def smiles_to_pyg_graph(smiles, y=None):
    """
    Convert a SMILES string to a PyTorch Geometric Data object with node and edge features.
    
    Args:
        smiles (str): SMILES string of the molecule
        y (float, optional): Target value for the molecule
    
    Returns:
        torch_geometric.data.Data: PyTorch Geometric Data object
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    # Get node features
    node_features = []
    for atom in mol.GetAtoms():
        node_features.append(get_atom_features(atom))
    node_features = np.array(node_features)
    
    # Get edge features and indices
    edge_index = []
    edge_features = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_index.extend([[i, j], [j, i]])  # Add both directions
        edge_features.extend([get_bond_features(bond)] * 2)  # Duplicate for both directions
    
    if not edge_index:  # Handle molecules with no bonds
        return None
    
    edge_index = np.array(edge_index).T
    edge_features = np.array(edge_features)
    
    # Convert to PyTorch tensors
    x = torch.tensor(node_features, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    edge_attr = torch.tensor(edge_features, dtype=torch.float)
    
    # Create PyTorch Geometric Data object
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr
    )
    
    if y is not None:
        data.y = torch.tensor([y], dtype=torch.float)
    
    return data

def convert_dataset_to_pyg(dataset):
    """
    Convert a list of (smiles, y) pairs to a list of PyTorch Geometric Data objects.
    
    Args:
        dataset (list): List of (smiles, y) pairs
    
    Returns:
        list: List of PyTorch Geometric Data objects
    """
    pyg_dataset = []
    for smiles, y in dataset:
        pyg_graph = smiles_to_pyg_graph(smiles, y)
        if pyg_graph is not None:
            pyg_dataset.append(pyg_graph)
    return pyg_dataset
