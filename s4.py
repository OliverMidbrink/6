from rdkit import Chem
from rdkit.Chem import Descriptors
import os
from mordred import Chi, ABCIndex, RingCount, Calculator, is_missing, descriptors

def get_mol_descriptors(smiles_input_str, calculator = Calculator(descriptors)):
    return calculator(Chem.MolFromSmiles(smiles_input_str))


descriptors = get_mol_descriptors("CC(C)C(=O)c1cc(C(=O)c2ccc(Oc3ccccc3)cc2)c(O)c(O)c1O")


from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, MACCSkeys

def generate_fingerprints(smiles):
    # Convert SMILES to RDKit molecule
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Generate PubChem-like fingerprint (approximation using Morgan)
    pubchem_fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=881)

    # Generate ECFP (Extended-Connectivity Fingerprints)
    ecfp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=2048)

    # Generate MACCS keys
    maccs_keys = MACCSkeys.GenMACCSKeys(mol)

    # Regenerate SMILES
    generated_smiles = Chem.MolToSmiles(mol)

    return pubchem_fp, ecfp, maccs_keys, generated_smiles

# Example SMILES string
smiles = 'CCO'  # Ethanol

# Generate fingerprints
fingerprints = generate_fingerprints(smiles)

if fingerprints:
    pubchem_fp, ecfp, maccs_keys, generated_smiles = fingerprints
    print("PubChem-like Fingerprint:", pubchem_fp)
    print("ECFP:", ecfp)
    print("MACCS Keys:", maccs_keys)
    print("Regenerated SMILES:", generated_smiles)
else:
    print("Invalid SMILES string.")
