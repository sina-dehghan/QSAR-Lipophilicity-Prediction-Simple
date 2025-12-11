import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from rdkit.Chem import MACCSkeys    # for generating the fingerprint

from sklearn.model_selection import train_test_split    # for spliting them in the two group of training and testing
from sklearn.ensemble import RandomForestRegressor      # Importing the ML algorithem based on the ensemble of decision trees
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb   # It's used for the gradient boosting; builds many decisions trees, each correcting erros of the previous ones.

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)       # Just becuase scikit-learn sometimes makes noisy warning


#   Phase 1: Loading data
# I am plaining to use the TDC dataset. (Therapeutics Data Commons)
def load_lipophilicity_data():
    print("=" * 75)
    print("STEP 1: Loading Lipophilicity Dataset")
    print("=" * 75)

    try:
        from tdc.single_pred import ADME    #it's library that provides clean, ready-to-use datasets. So I don't have to clean datas too.
        # ADME: Absorption, Distribution, Metabolism, Excretion properties
        """example of tdc:
        from tdc.single_pred import ADME
        data = ADME(name = 'Caco2_Wang')
        df = data.get_data()
        splits = data.get_split()
        """
        print("Downloading dataset from Therapeutics Data Commons...")

        data = ADME(name='Lipophilicity_AstraZeneca')
        df = data.get_data()

        print(f"✓ Dataset loaded successfully!")
        print(f"  - Total molecules: {len(df)}")
        print(f"  - Columns: {list(df.columns)}")
        print(f"\nFirst few molecules: {df.head()}")
        print(f"\nLogP statistics:")
        print(df['Y'].describe())
        
        return df
    except ImportError:
        print("TDC not installed. Install with: pip install PyTDC")
        raise

# Phase 2: Molecular Descriptor Generation

def smiles_to_mol(smiles):
    # Converting SMILES string to RDKit molecule object.

    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol
    except:
        return None


def generate_ecfp4_fingerprints(smiles_list):
    # Generate ECFP4 (Extended Connectivity Fingerprints) for molecules.
    print("\nGenerating ECFP4 Fingerprints...")
    fingerprints = []

    for smiles in smiles_list:
        mol = smiles_to_mol(smiles)
        if mol is not None:
            # Generate Morgan fingerprint (ECFP) with radius 2 (=ECFP4)
            fp = AllChem.GetMorganFinferprintAsBitVect(mol, radius=2, nBits=2048)   # radius is 2 since, 2x2=4 as we had for the ECFP4 and our fingerprint is 2048-bit vector (a long list of 0s and 1s)

            # Convert to numpy array
            arr = np.zeros((1,))
            fingerprints.append(np.array(fp))
        else:
            # If molecule is invalid, use zeros
            fingerprints.append(np.zeros(2048))
        
    print(f"  ✓ Generated {len(fingerprints)} ECFP4 fingerprints (2048 bits each)")
    return np.array(fingerprints)


def generate_maccs_keys(smiles_list):
    # Generate MACCS (Molecular ACCess System) keys for molecules; a fixed set of 166 yes/no (0/1) questions about a molecule.
    # it's good for the quick similarity checking

    print("\nGenerating MACCS Keys...")
    maccs_list = []

    for smiles in smiles_list:
        mol = smiles_to_mol(smiles)
        if mol is not None:
            # Generate MACCS keys (167 bits, but first is always 0)
            maccs = MACCSkeys.GenMACCSKeys(mol)
            maccs_list.append(np.array(maccs))
        else:
            maccs_list.append(np.zeros(164))

    print(f"  ✓ Generated {len(maccs_list)} MACCS fingerprints (167 bits each)")
    return np.array(maccs_list)

def generate_rdkit_descriptors(smiles_list):
    # Generating the RDkit Descriptors to calculate the properties. 
    print("\nGenerating RDKit Descriptors...")

    # creating a subset of usful descriptors
    descriptor_names = [
        'MolWt', 'MolLogP', 'NumHDonors', 'NumHAcceptors',
        'TPSA', 'NumRotatableBonds', 'NumAromaticRings',
        'NumAliphaticRings', 'FractionCSP3', 'HeavyAtomCount',
        'NumHeteroatoms', 'RingCount', 'MolMR'
    ]
    descriptor_data = []

    for smiles in smiles_list:
        mol = smiles_to_mol(smiles)
        if mol is not None:
            desc_values = []
            for desc_name in descriptor_names:
                try:    #getting the descriptor function form RDkit
                    desc_func = getattr(Descriptors, desc_name)
                    value = desc_func(mol)
                    desc_values.append(value)
                except:
                    desc_values.append(0)
                descriptor_data.append(desc_values)
        else:
            descriptor_data.append([0] * len(descriptor_names))
    
    print(f"  ✓ Generated {len(descriptor_names)} RDKit descriptors for {len(smiles_list)} molecules")
    print(f"  Descriptors used: {', '.join(descriptor_names)}")

    return np.array(descriptor_data)


def generate_combined_descriptors(smiles_list):
    # Combining all the descriptors; (ECFP4+MACCS+RDKit)
    print("\nGenerating Combined Descriptors (ECFP4 + MACCS + RDKit)...")

    ecfp4 = generate_ecfp4_fingerprints(smiles_list)
    maccs = generate_maccs_keys(smiles_list)
    rdkit_desc = generate_rdkit_descriptors(smiles_list)

    combined = np.hstack([ecfp4, maccs, rdkit_desc])

    print(f"  ✓ Combined descriptor shape: {combined.shape}")
    print(f"    (2048 ECFP4 + 167 MACCS + 13 RDKit = {combined.shape[1]} total features)")
    
    return combined

# Phase 3: Machine Learning Model

def train_random_forest(X_train, y_train, X_test, y_test):  # Training Random Forest regression model.
    print("\n" + "=" * 80)
    print("Training Random Forest Model")
    print("=" * 80)

    # Creating the model with 100 trees
    model = RandomForestRegressor(
        n_estimators=100,       # Number of trees
        max_depth=20,           # Maximum depth of each tree
        min_samples_split=5,    # Minimum samples to split a node
        random_state=42,        # for reproducibility
        n_jobs=-1,              # Use all CPU cores
    )

    # Train the model
    print("(RandomForest)Training...")
    model.fit(X_train, y_train)

    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Calculate performance metrics
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)

    print(f"\n✓ Training Complete!")
    print(f"\nTraining Set Performance:")
    print(f"  - R² Score: {train_r2:.4f}")
    print(f"  - RMSE: {train_rmse:.4f}")
    print(f"  - MAE: {train_mae:.4f}")
    print(f"\nTest Set Performance:")
    print(f"  - R² Score: {test_r2:.4f}")
    print(f"  - RMSE: {test_rmse:.4f}")
    print(f"  - MAE: {test_mae:.4f}")

    return {
        'model': model,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'test_mae': test_mae
    }


def train_xgboost(X_train, y_train, X_test, y_test):    # Train an XGBoost regreesion model.
    # Xgboost is a gradient boosting model.
    # Building tree, which correct the previous tree    # Random forest is independent (we want to fix the previous mistakes)
    
    print("\n" + "=" * 80)
    print("Training XGBoost Model")
    print("=" * 80)

    # Create model
    model = xgb.XGBRegressor(
        n_estimators=100,       # Number of boosting rounds
        max_depth=6,            # Maximum depth of tree
        learning_rate=0.1,      # Step size for each tree
        subsample=0.8,          # Fraction of samples for each tree
        colsample_bytree=0.8,    # Fractino of features for each tree
        random_state=42
    )

    # Train the model
    print("(XgBoost)Training...")
    model.fit(X_train, y_train)

    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    













