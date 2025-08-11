import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Draw, PandasTools, Descriptors, Lipinski

# Set up plotting
import deepchem as dc


np.random.seed(123)


n_features = 1024

tox21_tasks, tox21_dataset, tox21_transformers = dc.molnet.load_tox21(
    split = 'random', featurizer = 'GraphConv'
)
tox21_train, tox21_valid, tox21_test = tox21_dataset

# Create a GraphConv model
model = dc.models.GCNModel(
    len(tox21_tasks),  # Number of tasks
    mode='classification',  # TOX21 is a classification task
    batch_size=32,
    learning_rate=0.001
)


