import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from rdkit.Chem import MACCSkeys
from rdkit.Chem import Chem, DataStructs

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xboost as xgb


import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

def load_lipophillicity_data():
    print("=" * 75)
    print("STEP 1: Loading Lipophilicity Dataset")
    print("=" * 75)

    try:
        from tdc.single_pred import ADME
        print("Downloading dataset form TDC...")
        data = ADME(name='Lipophilicity_AstraZeneca')
        df = data. get_data()

        return df
    except ImportError:
        print("TDC not installed.")


def smiles_to_mol(smiles):

    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol
    except:
        return None


def generate_ecfp4_fingerprints(smiles_list):
    fingerprints = []

    for smiles in smiles_list:
        mol = smiles_to_mol(smiles)
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
            arr = np.zeros((2048,))
            DataStructs.ConvertToNumpyArray(fp, arr)
            fingerprints.append (arr)
        else:
            fingerprints.append(np.zeros(2048))

def generate_maccs_keys(smiles_list):
    maccs_list = []

    for smiles in smiles_list:
        mol = smiles_to_mol(smiles)
        if mol is not None:
            maccs = MACCSkeys.GenMACCSKeys(mol)
            arr = np.zeros((167,))
            DataStructs.ConvertToNumpyArray(maccs, arr)
            maccs_list.append(arr)
        else:
            maccs_list.append(np.zeros(167))
    return np.array(maccs_list)

def generate_rdkit_descriptors(smiles_list):
    descriptor_names = ['MolWt', 'MolLogP', 'NumHDonors', 'NumHAcceptors',
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
                try:
                    desc_func = getattr(Descriptors, desc_name)
                    value = desc_func(mol)
                    desc_values.append(value)
                except:
                    desc_values.append(0)
            descriptor_data.append(desc_values)
        else:
            descriptor_data.append([0] * len(descriptor_names))
    return np.array(descriptor_data)

def generate_combined_descriptors(smiles_list):
    ecfp4 = generate_ecfp4_fingerprints(smiles_list)
    maccs = generate_maccs_keys(smiles_list)
    rdkit_desc = generate_rdkit_descriptors(smiles_list)

    combined = np.hstack([ecfp4, maccs, rdkit_desc])
    return combined


def train_random_forest(X_train, y_train, X_test, y_test):
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)

    return {
        'model': model,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'test_mae': test_mae
    }

def train_xgboost(X_train,y_train, X_test, y_test):
    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)

    return {
        'model': model,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_mae': train_mae,
        'test_mae': test_mae
    }

def compare_descriptors_and_models(df):
    results = []

    smiles_list =df['Drug'].tolist()
    y = df ['Y'].values

    descriptor_types = {
        'ECFP4': generate_ecfp4_fingerprints,
        'MACCS': generate_maccs_keys,
        'RDKit': generate_rdkit_descriptors,
        'Combined': generate_combined_descriptors
    }

    for desc_name, desc_func in descriptor_types.items():
        X = desc_func(smiles_list)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        rf_result = train_random_forest(X_train, y_train, X_test, y_test)
        results.append({
            'Descriptor': desc_name,
            'Model': 'Random Forest',
            'Test R²': rf_results['test_r2'],
            'Test RMSE': rf_results['test_rmse'],
            'Test MAE': rf_results['test_mae']
        })

        xgb_results = train_xgboost(X_train,y_train,X_test,y_test)
        results.append({
            'Descriptor': desc_name,
            'Model': 'XGBoost',
            'Test R²': xgb_results['test_r2'],
            'Test RMSE': xgb_results['test_rmse'],
            'Test MAE': xgb_results['test_mae']
        })

        results_df = pd.DataFrame(results)
        