from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, MolSurf, Lipinski, rdMolDescriptors
import pandas as pd

# === Feature Extraction Function ===
def extract_descriptors(smiles_list):
    data = []

    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            desc = {
                # --- 3.3.1 Molecular Descriptors ---
                'MolWt': Descriptors.MolWt(mol),
                'LogP': Crippen.MolLogP(mol),
                'TPSA': MolSurf.TPSA(mol),
                'NumHDonors': Lipinski.NumHDonors(mol),
                'NumHAcceptors': Lipinski.NumHAcceptors(mol),
                'NumRotatableBonds': Lipinski.NumRotatableBonds(mol),

                # --- 3.3.2 Topological Descriptors ---
                'Chi0': Descriptors.Chi0n(mol),
                'Chi1': Descriptors.Chi1n(mol),
                'Kappa1': Descriptors.Kappa1(mol),
                'WienerIndex': Descriptors.WienerIndex(mol),
                'BalabanJ': Descriptors.BalabanJ(mol),

                # --- 3.3.3 Placeholder for Interaction-based Descriptors ---
                'DockingScore': -7.5,  # Placeholder (from molecular docking)
                'HydrogenBonds': 3,   # Placeholder
            }
        else:
            desc = {k: 0 for k in [
                'MolWt', 'LogP', 'TPSA', 'NumHDonors', 'NumHAcceptors', 'NumRotatableBonds',
                'Chi0', 'Chi1', 'Kappa1', 'WienerIndex', 'BalabanJ', 'DockingScore', 'HydrogenBonds'
            ]}
        data.append(desc)

    return pd.DataFrame(data)
