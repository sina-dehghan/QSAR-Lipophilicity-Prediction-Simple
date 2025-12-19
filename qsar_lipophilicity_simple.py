import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs                              # We need this for the finger prent conversations
from rdkit.Chem import Descriptors, AllChem, MACCSkeys           # for generating the fingerprint
                             
import joblib                                                    # for saving best model (Chat-GPT)

from sklearn.model_selection import train_test_split             # for spliting them in the two group of training and testing
from sklearn.ensemble import RandomForestRegressor               # Importing the ML algorithem based on the ensemble of decision trees
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb                                            # It's used for the gradient boosting; builds many decisions trees, each correcting erros of the previous ones.

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)        # Just becuase scikit-learn sometimes makes noisy warning


#   Phase 1: Loading data
# I am plaining to use the TDC dataset. (Therapeutics Data Commons)
def load_lipophilicity_data():
    print("=" * 75)
    print("STEP 1: Loading Lipophilicity Dataset")
    print("=" * 75)

    try:
        from tdc.single_pred import ADME            # The error is for the library; it's okay.
        #it's library that provides clean, ready-to-use datasets. So I don't have to clean datas too.
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

# New: Validating dataset and checking the right columns that we need. (from Chat-GPT; must Check.)
def validate_dataset(df):
    
    print("\nValidating dataset...")
    
    # Check required columns
    required_cols = ['Drug', 'Y']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Check for empty dataset
    if len(df) == 0:
        raise ValueError("Dataset is empty!")
    
    # Check for minimum samples
    if len(df) < 100:
        print(f"WARNING: Only {len(df)} samples. Need at least 100 for reliable results.")
    
    # Check for missing values
    null_counts = df.isnull().sum()
    if null_counts.any():
        print(f"WARNING: Found missing values:\n{null_counts[null_counts > 0]}")
    
    # Check for invalid SMILES
    invalid_count = 0
    for smiles in df['Drug'][:10]:  # Check first 10
        if smiles_to_mol(smiles) is None:
            invalid_count += 1
    
    if invalid_count > 0:
        print(f"WARNING: Found {invalid_count}/10 invalid SMILES in sample")
    
    print("✓ Dataset validation complete")
    return True
# New

# Phase 2: Molecular Descriptor Generation

def smiles_to_mol(smiles):
    # Converting SMILES string to RDKit molecule object.

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"Warning: Invalid SMILES string: {smiles}")
        return mol
    except Exception as e:
        print(f"Error processing SMILES '{smiles}': {str(e)}")
        return None


def generate_ecfp4_fingerprints(smiles_list):
    # Generate ECFP4 (Extended Connectivity Fingerprints) for molecules.
    print("\nGenerating ECFP4 Fingerprints...")
    fingerprints = []

    for smiles in smiles_list:
        mol = smiles_to_mol(smiles)
        if mol is not None:
            # Generate Morgan fingerprint (ECFP) with radius 2 (=ECFP4)
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)   # radius is 2 since, 2x2=4 as we had for the ECFP4 and our fingerprint is 2048-bit vector (a long list of 0s and 1s)

            # Convert to numpy array
            arr = np.zeros((2048,))             # Size
            DataStructs.ConvertToNumpyArray(fp, arr)
            fingerprints.append(arr)
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
            # The point is that 0 is unused; so, 0 unused + 166 bits
            maccs = MACCSkeys.GenMACCSKeys(mol)
            arr = np.zeros((167,))
            DataStructs.ConvertToNumpyArray(maccs, arr)
            maccs_list.append(arr)
        else:
            maccs_list.append(np.zeros(167))
            # Using the invalid fp for the cases that the mol is None or SMILES is invalid.

    print(f"  ✓ Generated {len(maccs_list)} MACCS fingerprints (167 bits each)")
    return np.array(maccs_list)

def generate_rdkit_descriptors(smiles_list):
    # Generating the RDkit Descriptors to calculate the properties. 
    print("\nGenerating RDKit Descriptors...")

    # creating a subset of usful descriptors
    descriptor_names = [
        'MolWt', 'NumHDonors', 'NumHAcceptors',                     # Removing MolLogP ,which is RDKit's prediction of LogP, to check the model.
        'TPSA', 'NumRotatableBonds', 'NumAromaticRings',
        'NumAliphaticRings', 'FractionCSP3', 'HeavyAtomCount',
        'NumHeteroatoms', 'RingCount', 'MolMR', 'NumSaturatedRings' # adding 'NumSaturatedRings'
    ]
    descriptor_data = []

    for smiles in smiles_list:
        mol = smiles_to_mol(smiles)
        if mol is not None:
            desc_values = []                                             # Descriptor values for one molecule.
            for desc_name in descriptor_names:
                try:    #getting the descriptor function form RDkit
                    desc_func = getattr(Descriptors, desc_name)         # Instead of Looping them all one by one we'll use "getattr"
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
        max_depth=20,           # Maximum depth of each tree (Reduced from 20)
        min_samples_split=10,    # Minimum samples to split a node (Increased from 5)
        min_samples_leaf=4,     # ← NEW: Minimum samples in leaves
        max_features='sqrt',    # ← NEW: Use sqrt of features per tree
        random_state=42,        # for reproducibility
        n_jobs=-1,              # Use all CPU cores, 1= single core
    )

    # Train the model
    print("(RandomForest)Training...")
    model.fit(X_train, y_train) 
    # Train = model LEARNS the patterns of the 80% of data
    # X_train will be the numerical features. (In this case it would be the 80% of dataset.)
    # y_train will be the result of the Descriptors. (In this case would be the lipophilicites of 80% of daataset.)
    
    # Test = If the model can predict values for the new molecules.
    # X_test will be the remaining 20% of the dataset.
    # y_test will be lipophilicity of those 20% data.
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
        n_estimators=100,           # Number of boosting rounds
        max_depth=4,                # Maximum depth of tree.    (Reduced from 6, which was deep and ovverfit)
        min_child_weight=3,         # New: Prevents tiny leaves that fit noise
        gamma=0.1,                  # New: Making model more conservative for spliting
        reg_alpha=0.1,              # New: L1 regularization. Penalize large feature weights
        reg_lambda=1.0,             # New: L2 regularization
        learning_rate=0.1,          # Step size for each tree
        subsample=0.8,              # Fraction of samples for each tree
        colsample_bytree=0.8,       # Fractino of features for each tree
        random_state=42
    )

    # Train the model
    print("(XgBoost)Training...")
    model.fit(X_train, y_train)

    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Calculate performace metrics
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
        'train_mae': train_mae,
        'test_mae': test_mae
    }


# Phase 4: Model Comparison
def compare_descriptors_and_models(df):
    # Compare different descriptor types and models.
    # We need to test each descriptor (ECFP4, MACCS, RDKit, Combined) + model (Random Forest, XGBoost)
    # We need to show which combination works best!
    print("\n" + "=" * 80)
    print("COMPARING ALL DESCRIPTOR TYPES AND MODELS")
    print("=" * 80)

    results = []

    # perpearing the data
    smiles_list = df['Drug'].tolist()
    y = df ['Y'].values             # Column Y is related to the lipophilicity

    # Dictionary of descriptor generators
    descriptor_types = {
        'ECFP4': generate_ecfp4_fingerprints,
        'MACCS': generate_maccs_keys,
        'RDKit': generate_rdkit_descriptors,
        'Combined': generate_combined_descriptors
    }

    # Test each descriptor type
    for desc_name, desc_func in descriptor_types.items():
        print(f"\n{'=' * 80}")
        print(f"Testing Descriptor Type: {desc_name}")
        print(f"{'=' * 80}")

        # Generate descriptors
        X = desc_func(smiles_list)
        
        # Split data: 80% training, 20% testing(test_size=0.2)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        print(f"\nDataset split:")
        print(f"  - Training samples: {len(X_train)}")
        print(f"  - Test samples: {len(X_test)}")

        # Train Random Forest
        rf_results = train_random_forest(X_train, y_train, X_test, y_test)
        results.append({
            'Descriptor': desc_name,
            'Model': 'Random Forest',
            'Test R²': rf_results['test_r2'],
            'Test RMSE': rf_results['test_rmse'],
            'Test MAE': rf_results['test_mae']
        })

        # Train XGBoost
        xgb_results = train_xgboost(X_train, y_train, X_test, y_test)
        results.append({
            'Descriptor': desc_name,
            'Model': 'XGBoost',
            'Test R²': xgb_results['test_r2'],
            'Test RMSE': xgb_results['test_rmse'],
            'Test MAE': xgb_results['test_mae']
        })

        # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    return results_df

# ==============
# Final Phase: MAIN EXECUTION
# ==============

def main(): 
# Main function that runs the entire QSAR pipeline.
    banner = r"""
   ___  ____    _    ____    __  __    _    ____ _____ _____ ____  
  / _ \/ ___|  / \  |  _ \  |  \/  |  / \  / ___|_   _| ____|  _ \ 
 | | | \___ \ / _ \ | |_) | | |\/| | / _ \ \___ \ | | |  _| | |_) |
 | |_| |___) / ___ \|  _ <  | |  | |/ ___ \ ___) || | | |___|  _ < 
  \__\_\____/_/   \_\_| \_\ |_|  |_/_/   \_\____/ |_| |_____|_| \_\
    
    >>> Developed by: Sina 
    >>> Version: 1.0 (2025)
    """
    
    print("\n" + "=" * 80)
    print(banner)                   # (if you like the banner search for "ASCII Art Generator")
    print("=" * 80)    




    print("\n" + "=" * 80)
    print("QSAR LIPOPHILICITY PREDICTION PIPELINE")
    print("=" * 80)
    print("\nThis program will:")
    print("1. Load lipophilicity dataset from TDC")
    print("2. Generate molecular descriptors (ECFP4, MACCS, RDKit, Combined)")
    print("3. Train machine learning models (Random Forest, XGBoost)")
    print("4. Compare all combinations and show results")
    print("\n")
    
    # Load data
    df = load_lipophilicity_data()

    # NEW: Calling Validate dataset
    validate_dataset(df)
    
    # Compare all descriptors and models
    results_df = compare_descriptors_and_models(df)
    
    # Display final results
    print("\n" + "=" * 80)
    print("FINAL RESULTS - ALL COMBINATIONS")
    print("=" * 80)
    print("\n", results_df.to_string(index=False))
    
    # Find best model
    best_idx = results_df['Test R²'].idxmax()
    best_result = results_df.iloc[best_idx]
    
    print("\n" + "=" * 80)
    print("BEST PERFORMING MODEL")
    print("=" * 80)
    print(f"\nDescriptor Type: {best_result['Descriptor']}")
    print(f"Model: {best_result['Model']}")
    print(f"Test R² Score: {best_result['Test R²']:.4f}")
    print(f"Test RMSE: {best_result['Test RMSE']:.4f}")
    print(f"Test MAE: {best_result['Test MAE']:.4f}")
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)
    
    # New (Chat-GPT)---------------------------------------
    # ... all existing code until the end ...
    
    print("\n" + "=" * 80)
    print("BEST PERFORMING MODEL")
    print("=" * 80)
    print(f"\nDescriptor Type: {best_result['Descriptor']}")
    print(f"Model: {best_result['Model']}")
    print(f"Test R² Score: {best_result['Test R²']:.4f}")
    print(f"Test RMSE: {best_result['Test RMSE']:.4f}")
    print(f"Test MAE: {best_result['Test MAE']:.4f}")
    
    # NEW: Retrain best model and save it
    print("\n" + "=" * 80)
    print("SAVING BEST MODEL")
    print("=" * 80)
    
    # Regenerate best descriptors
    smiles_list = df['Drug'].tolist()
    y = df['Y'].values
    
    if best_result['Descriptor'] == 'ECFP4':
        X = generate_ecfp4_fingerprints(smiles_list)
    elif best_result['Descriptor'] == 'MACCS':
        X = generate_maccs_keys(smiles_list)
    elif best_result['Descriptor'] == 'RDKit':
        X = generate_rdkit_descriptors(smiles_list)
    else:  # Combined
        X = generate_combined_descriptors(smiles_list)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Retrain best model
    if best_result['Model'] == 'XGBoost':
        final_model = train_xgboost(X_train, y_train, X_test, y_test)['model']
    else:
        final_model = train_random_forest(X_train, y_train, X_test, y_test)['model']
    
    # Save model
    model_filename = f"best_model_{best_result['Descriptor']}_{best_result['Model'].replace(' ', '_')}.pkl"
    joblib.dump(final_model, model_filename)
    print(f"✓ Model saved to '{model_filename}'")
    
    # Save descriptor type info
    with open('model_descriptor_type.txt', 'w') as f:
        f.write(f"Descriptor: {best_result['Descriptor']}\n")
        f.write(f"Model: {best_result['Model']}\n")
        f.write(f"Test R²: {best_result['Test R²']:.4f}\n")
    print("✓ Model metadata saved to 'model_descriptor_type.txt'")
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)
    # New (Chat-GPT)--------------------------------------------

    # Save results
    results_df.to_csv('qsar_results.csv', index=False)
    print("\n✓ Results saved to 'qsar_results.csv'")
    
    return results_df, final_model  #New final_model added


if __name__ == "__main__":
   
    results = main()










