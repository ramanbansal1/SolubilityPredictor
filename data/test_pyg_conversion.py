import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyg_converter import smiles_to_pyg_graph
from rdkit import Chem
import torch
import numpy as np

# Test with a simple molecule
smiles = 'CCO'  # Ethanol
y = 0.5  # Example target value

# Convert to PyG graph
data = smiles_to_pyg_graph(smiles, y)

if data is not None:
    print("\nPyTorch Geometric Data object created successfully!")
    print("\nNode Features (x):")
    print(data.x)
    print("\nEdge Index:")
    print(data.edge_index)
    print("\nEdge Features (edge_attr):")
    print(data.edge_attr)
    print("\nTarget (y):")
    print(data.y)
    
    # Verify the number of nodes and edges
    num_nodes = data.x.size(0)
    num_edges = data.edge_index.size(1) // 2  # Divide by 2 because edges are bidirectional
    print(f"\nNumber of nodes: {num_nodes}")
    print(f"Number of edges: {num_edges}")
    
    # Verify the feature dimensions
    print(f"\nNode feature dimension: {data.x.size(1)}")
    print(f"Edge feature dimension: {data.edge_attr.size(1)}")
else:
    print("Failed to create PyTorch Geometric Data object")
