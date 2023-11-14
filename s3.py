from rdkit import Chem
from rdkit.Chem import Descriptors

# Define a function to convert a SMILES string to RDKit molecular descriptors
def smiles_to_descriptors(smiles_string):
    mol = Chem.MolFromSmiles(smiles_string)

    if mol is not None:
        descriptors = {
            "Molecular Weight": Descriptors.MolWt(mol),
            "LogP": Descriptors.MolLogP(mol),
            "Number of Hydrogen Bond Donors": Descriptors.NumHDonors(mol),
            "Number of Hydrogen Bond Acceptors": Descriptors.NumHAcceptors(mol),
            "Polar Surface Area": Descriptors.TPSA(mol),
        }
        return descriptors
    else:
        return None

# Example usage
smiles = "CC(C)C(=O)c1cc(C(=O)c2ccc(Oc3ccccc3)cc2)c(O)c(O)c1O"  # Replace with your SMILES string
descriptors = smiles_to_descriptors(smiles)

if descriptors is not None:
    print("Molecular Descriptors:")
    for descriptor, value in descriptors.items():
        print(f"{descriptor}: {value}")
else:
    print("Invalid SMILES string.")
