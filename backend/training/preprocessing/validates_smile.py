import pandas as pd
from rdkit import Chem
from pathlib import Path

# =========================
# CONFIG
# =========================

INPUT_PATH = Path("data/raw/dataset.csv")
OUTPUT_PATH = Path("data/processed/clean_smiles.csv")

SMILES_COLUMN = "smiles"  # adapte si le nom change


# =========================
# FUNCTIONS
# =========================

def is_valid_smiles(smiles: str) -> bool:
    """
    Vérifie si un SMILES est valide chimiquement.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except:
        return False


def canonicalize_smiles(smiles: str) -> str:
    """
    Convertit un SMILES en forme canonique.
    """
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol, canonical=True)


# =========================
# MAIN
# =========================

def main():
    print("Loading dataset...")

    df = pd.read_csv(INPUT_PATH)

    print(f"Initial molecules: {len(df)}")

    # Suppression des NaN
    df = df.dropna(subset=[SMILES_COLUMN])

    # Validation
    df["is_valid"] = df[SMILES_COLUMN].apply(is_valid_smiles)

    valid_df = df[df["is_valid"]].copy()

    print(f"Valid molecules: {len(valid_df)}")

    # Canonicalisation
    valid_df["canonical_smiles"] = valid_df[SMILES_COLUMN].apply(
        canonicalize_smiles
    )

    # Suppression doublons
    valid_df = valid_df.drop_duplicates(subset=["canonical_smiles"])

    print(f"Unique molecules: {len(valid_df)}")

    # Sauvegarde
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    valid_df.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved cleaned dataset to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
