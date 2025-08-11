import os 
import requests
import pandas as pd

url = "https://raw.githubusercontent.com/ESAIBMCIC/MolecularSolubilityPrediction/master/delaney.csv"
df = pd.read_csv(url)
print(df.head())
