"""
In this file, I am providing you the need visualization that can help us
to undestant this program better.

Author: Sina Dehghan + helps of our dear ChatGPT :D
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import joblib   # For loading saved models
import pickle   # For loading saved data
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, MACCSkeys, DataStructs
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

# Set styl for better looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10

# Phase 1: Data Distribution Visualizations
# We use these for understaing our dataset before modeling

def plot_lipophilicity_distribution(df, save_path='plots/lipophilicity_distribution.png'):
    # df: DataFrame with 'Y' column which is for the lipophilicity

    print("\n📊 Creating lipophilicity distribution plot...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Histogram
    axes[0].hist(df['Y'], bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('LogP (Lipophilicity)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Number of Molecules', fontsize=12, fontweight='bold')
    axes[0].set_title('Distribution of Lipophilicity Values', fontsize=14, fontweight='bold')
    axes[0].axvline(df['Y'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df["Y"].mean():.2f}')
    axes[0].axvline(df['Y'].median(), color='green', linestyle='--', linewidth=2, label=f'Median: {df["Y"].median():.2f}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Box Plot (for showing outliers and quartiles)
    axes[1].boxplot(df['Y'], vert=True, patch_artist=True,
                    boxprops=dict(facecolor='lightblue', color='steelblue'),
                    medianprops=dict(color='red', linewidth=2),
                    whiskerprops=dict(color='steelblue'),
                    capprops=dict(color='steelblue'))
    axes[1].set_ylabel('LogP (Lipophilicity)', fontsize=12, fontweight='bold')
    axes[1].set_title('Box Plot: Identifying Outliers', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved to: {save_path}")
    plt.close()

def plot_molecular_property_correlations(df, save_path='plots/property_correlations.png'):
    # showing which molecular properties are related to each other
    # Better identifying redundant features
    # Red = Strong Positve Correlation, Blue = Strong Negative Correlation

    print("\n📊 Creating molecular property correlation heatmap...")

    # Calculate descriptors for a sample
    sample_size = min(500, len(df)) # Use 500 molecules or less
    sample_df = df.sample(n=sample_size, random_state=42)

    descriptor_names = [
        'MolWt', 'NumHDonors', 'NumHAcceptors',                     # Removing MolLogP ,which is RDKit's prediction of LogP, to check the model.
        'TPSA', 'NumRotatableBonds', 'NumAromaticRings',
        'NumAliphaticRings', 'FractionCSP3', 'HeavyAtomCount',
        'NumHeteroatoms', 'RingCount', 'MolMR', 'NumSaturatedRings' # adding 'NumSaturatedRings'
    ]

    desc_data = []
    for smiles in sample_df['Drug']:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            desc_values = [getattr(Descriptors, name)(mol) for name in descriptor_names]
            desc_data.append(desc_values)
    
    desc_df = pd.DataFrame(desc_data, columns=descriptor_names)

    # Calculate correlation matrix
    corr_matrix = desc_df.corr()

    # Create Heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=1,
                cbar_kws={'label': 'Correlation Coefficient'})
    plt.title('Correlation Between Molecular Descriptors', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved to: {save_path}")
    plt.close()


# Phase 2: Model Performace Visualizations
# These will help us compare different models and descriptors
def plot_model_comparison_bars(results_df, save_path='plots/model_comparison.png'):
    # Creating bar charts comparing all model/descriptor combinations

    print("\n📊 Creating model comparison bar charts...")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Create x-axis lables(descriptor + model)
    results_df['Label'] = results_df['Descriptor'] + '\n' + results_df['Model']
    x_pos = np.arange(len(results_df))

    # Color Palette for different descriptors
    colors = {'ECFP4': '#FF6B6B', 'MACCS': '#4ECDC4', 
              'RDKit': '#45B7D1', 'Combined': '#FFA07A'}
    bar_colors = [colors[desc] for desc in results_df['Descriptor']]
    
    # Plot 1:R² Score (higher is better)
    axes[0].bar(x_pos, results_df['Test R²'], color=bar_colors, edgecolor='black', linewidth=1.5)
    axes[0].set_ylabel('R² Score', fontsize=12, fontweight='bold')
    axes[0].set_title('R² Score Comparison\n(Higher is Better)', fontsize=14, fontweight='bold')
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(results_df['Label'], rotation=45, ha='right', fontsize=9)
    axes[0].axhline(y=0.8, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Good (0.8)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')

    # Plot 2: RMSE (lower is better)
    axes[1].bar(x_pos, results_df['Test RMSE'], color=bar_colors, edgecolor='black', linewidth=1.5)
    axes[1].set_ylabel('RMSE', fontsize=12, fontweight='bold')
    axes[1].set_title('RMSE Comparison\n(Lower is Better)', fontsize=14, fontweight='bold')
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(results_df['Label'], rotation=45, ha='right', fontsize=9)
    axes[1].grid(True, alpha=0.3, axis='y')

    # Plot 3: MAE (lower is better)
    axes[2].bar(x_pos, results_df['Test MAE'], color=bar_colors, edgecolor='black', linewidth=1.5)
    axes[2].set_ylabel('MAE', fontsize=12, fontweight='bold')
    axes[2].set_title('MAE Comparison\n(Lower is Better)', fontsize=14, fontweight='bold')
    axes[2].set_xticks(x_pos)
    axes[2].set_xticklabels(results_df['Label'], rotation=45, ha='right', fontsize=9)
    axes[2].grid(True, alpha=0.3, axis='y')

    # Add legend for descriptor colors
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color, edgecolor='black', label=desc) 
                      for desc, color in colors.items()]
    fig.legend(handles=legend_elements, loc='upper center', ncol=4, 
              bbox_to_anchor=(0.5, 0.98), fontsize=11, title='Descriptor Type')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved to: {save_path}")
    plt.close()

def plot_metric_heatmap(results_df, save_path='plots/metric_heatmap.png'):
    # Showing all metrics across models or descriptors by a heatmap.
    print("\n📊 Creating metric heatmap...")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    metrics = ['Test R²', 'Test RMSE', 'Test MAE']
    titles = ['R² Score (Higher is Better)', 'RMSE (Lower is Better)', 'MAE (Lower is Better)']
    cmaps = ['Greens', 'Reds_r', 'Reds_r']  # _r means reversed colormap
    
    for idx, (metric, title, cmap) in enumerate(zip(metrics, titles, cmaps)):
        # Pivot data for heatmap
        pivot_data = results_df.pivot(index='Model', columns='Descriptor', values=metric)

        # Create heatmap
        sns.heatmap(pivot_data, annot=True, fmt='.4f', cmap=cmap, ax=axes[idx],
                    cbar_kws={'label': metric}, linewidths=2, linecolor='black')
        axes[idx].set_title(title, fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Descriptor Type', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved to: {save_path}")
    plt.close()


# Phase 3: Prediction Quality Visualization
def plot_predicted_vs_actual(df, descriptor_type='Combined', model_type='XGBoost', 
                             save_path='plots/predicted_vs_actual.png'):
    # Create scatter plot of Predicted vs Actual lipophilicity values.
    print(f"\n📊 Creating predicted vs actual plot ({descriptor_type} + {model_type})...")

    # NEW: Load the saved model and test data
    try:
        clean_model_name = model_type.replace(' ', '_')
        base_name = f"{descriptor_type}_{clean_model_name}"

        # Load model
        model_path = f"saved_models/{base_name}_model.pkl"
        model = joblib.load(model_path)
        print(f"  ✓ Loaded model from {model_path}")

        # Load test data
        data_path = f"saved_models/{base_name}_test_data.npz"
        data = np.load(data_path)
        X_test = data['X_test']
        y_test = data['y_test']
        y_test_pred = data['y_test_pred']
        print(f"  ✓ Loaded test data from {data_path}")

        # Also get training predictions for the plot
        # We need to regenerate training data
        smiles_list = df['Drug'].tolist()
        y = df['Y'].values

        # Generate features based on descriptor type
        if descriptor_type == 'ECFP4':
            X = generate_ecfp4_fingerprints(smiles_list)
        elif descriptor_type == 'MACCS':
            X = generate_maccs_keys(smiles_list)
        elif descriptor_type == 'RDKit':
            X = generate_rdkit_descriptors(smiles_list)
        else:  # Combined
            X = generate_combined_descriptors(smiles_list)

        # Split data (same random_state=42 gives same split)
        from sklearn.model_selection import train_test_split
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Get training predictions
        y_train_pred = model.predict(X_train)

    except FileNotFoundError as e:
        print(f"  ✗ ERROR: Could not find saved model/data!")
        print(f"    {str(e)}")
        print(f"  ➜ Make sure you've run the main QSAR script first!")
        return None, None, None, None

    # Calculate metrics
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)


    # Create plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Training Set
    axes[0].scatter(y_train, y_train_pred, alpha=0.5, s=30, color='steelblue', edgecolors='navy')
    axes[0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 
                 'r--', lw=2, label='Perfect Prediction')
    axes[0].set_xlabel('Actual LogP', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Predicted LogP', fontsize=12, fontweight='bold')
    axes[0].set_title(f'Training Set: {descriptor_type} + {model_type}', fontsize=13, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Add metrics text box
    textstr = f'R² = {train_r2:.4f}\nRMSE = {train_rmse:.4f}\nMAE = {train_mae:.4f}'
    props = dict(boxstyle='round', facecolor='lightgreen', alpha=0.8)
    axes[0].text(0.05, 0.95, textstr, transform=axes[0].transAxes, fontsize=11,
                verticalalignment='top', bbox=props)

    # Plot 2: Test Set
    axes[1].scatter(y_test, y_test_pred, alpha=0.6, s=30, color='coral', edgecolors='darkred')
    axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                 'r--', lw=2, label='Perfect Prediction')
    axes[1].set_xlabel('Actual LogP', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Predicted LogP', fontsize=12, fontweight='bold')
    axes[1].set_title(f'Test Set: {descriptor_type} + {model_type}', fontsize=13, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Add metrics text box
    textstr = f'R² = {test_r2:.4f}\nRMSE = {test_rmse:.4f}\nMAE = {test_mae:.4f}'
    props = dict(boxstyle='round', facecolor='lightcoral', alpha=0.8)
    axes[1].text(0.05, 0.95, textstr, transform=axes[1].transAxes, fontsize=11,
                verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved to: {save_path}")
    plt.close()

    return model, X_test, y_test, y_test_pred

def plot_residuals(y_test, y_test_pred, descriptor_type='Combined', model_type='XGBoost',
                   save_path='plots/residual_plot.png'):
    # Ploting residuals or prediction errors will help us for checking the patterns.

    print(f"\n📊 Creating residual plot ({descriptor_type} + {model_type})...")

    residuals = y_test - y_test_pred

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Residuals vs Predicted
    axes[0].scatter(y_test_pred, residuals, alpha=0.6, s=50, color='purple', edgecolor='black', linewidth=0.5)
    axes[0].axhline(y=0, color='red', linestyle='--', linewidth=2)
    axes[0].set_xlabel('Predicted LogP', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Residuals (Actual - Predicted)', fontsize=12, fontweight='bold')
    axes[0].set_title(f'Residual Plot: {descriptor_type} + {model_type}', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].text(0.05, 0.95, 'Good: Random scatter around zero\nBad: Clear patterns', 
                transform=axes[0].transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # Plot 2: Histogram of residuals
    axes[1].hist(residuals, bins=30, color='orange', edgecolor='black', alpha=0.7)
    axes[1].axvline(x=0, color='red', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Residuals', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Frequency', fontsize=12, fontweight='bold')
    axes[1].set_title('Distribution of Residuals', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].text(0.05, 0.95, f'Mean: {residuals.mean():.4f}\nStd: {residuals.std():.4f}', 
                transform=axes[1].transAxes, fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved to: {save_path}")
    plt.close()

# Phase 4: Feature Importance Visualizations

def plot_feature_importance_rdkit(df, model_type='Random Forest', top_n=15,
                                 save_path='plots/feature_importance_rdkit.png'):
    
    print(f"\n📊 Creating feature importance plot (RDKit descriptors with {model_type})...")

    # NEW: Load the saved RDKit model
    try:
        clean_model_name = model_type.replace(' ', '_')
        base_name = f"RDKit_{clean_model_name}"
        model_path = f"saved_models/{base_name}_model.pkl"
        model = joblib.load(model_path)
        print(f"  ✓ Loaded model from {model_path}")
    except FileNotFoundError:
        print(f"  ✗ ERROR: Could not find {model_path}!")
        print(f"  ➜ Make sure you've run the main QSAR script first!")
        return

    # Descriptor names (must match what's in the main script)
    descriptor_names = [
        'MolWt', 'NumHDonors', 'NumHAcceptors',                     # Removing MolLogP ,which is RDKit's prediction of LogP, to check the model.
        'TPSA', 'NumRotatableBonds', 'NumAromaticRings',
        'NumAliphaticRings', 'FractionCSP3', 'HeavyAtomCount',
        'NumHeteroatoms', 'RingCount', 'MolMR', 'NumSaturatedRings' # adding 'NumSaturatedRings'
    ]
    
    # Get feature importance from loaded model
    importances = model.feature_importances_

    # Create DataFrame and sort
    importance_df = pd.DataFrame({
        'Feature': descriptor_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False).head(top_n)
    
    # Plot
    plt.figure(figsize=(12, 8))
    colors = plt.cm.viridis(np.linspace(0, 1, len(importance_df)))
    bars = plt.barh(range(len(importance_df)), importance_df['Importance'], color=colors, edgecolor='black', linewidth=1.5)
    plt.yticks(range(len(importance_df)), importance_df['Feature'])
    plt.xlabel('Importance Score', fontsize=12, fontweight='bold')
    plt.ylabel('Molecular Descriptor', fontsize=12, fontweight='bold')
    plt.title(f'Top {top_n} Most Important Features\n({model_type} Model)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='x')
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, importance_df['Importance'])):
        plt.text(value, i, f' {value:.4f}', va='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved to: {save_path}")
    plt.close()

# New Chat GPT
def plot_best_model_results(df):
    """
    Automatically load and plot results from the best performing model.
    """
    try:
        # Load best model info
        with open('best_model_info.pkl', 'rb') as f:
            best_info = pickle.load(f)
        
        descriptor = best_info['descriptor']
        model = best_info['model']
        
        print(f"\n🏆 Plotting best model: {descriptor} + {model}")
        print(f"   Test R² = {best_info['test_r2']:.4f}")
        
        # Create plots for best model
        model_obj, X_test, y_test, y_test_pred = plot_predicted_vs_actual(
            df, descriptor_type=descriptor, model_type=model
        )
        
        if model_obj is not None:
            plot_residuals(y_test, y_test_pred, 
                          descriptor_type=descriptor, model_type=model)
        
        return best_info
        
    except FileNotFoundError:
        print("  ✗ ERROR: best_model_info.pkl not found!")
        print("  ➜ Using default: Combined + XGBoost")
        
        model_obj, X_test, y_test, y_test_pred = plot_predicted_vs_actual(
            df, descriptor_type='Combined', model_type='XGBoost'
        )
        
        if model_obj is not None:
            plot_residuals(y_test, y_test_pred, 
                          descriptor_type='Combined', model_type='XGBoost')
        
        return None
# New Chat GPT

# =================================
# WHAT WE NEED FROM THE MAIN SCRIPT:
# =================================
def smiles_to_mol(smiles):
    try:
        return Chem.MolFromSmiles(smiles)
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
            fingerprints.append(arr)
        else:
            fingerprints.append(np.zeros(2048))
    return np.array(fingerprints)


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

# Main Visualization function
def create_all_visualizations():
    print("\n" + "=" * 80)
    print("QSAR VISUALIZATION SUITE")
    print("=" * 80)
    print("\nThis will create comprehensive visualizations for your QSAR analysis.")
    print("All plots will be saved in the 'plots/' directory.\n")
    
    # Create plots directory
    import os
    if not os.path.exists('plots'):
        os.makedirs('plots')
        print("✓ Created 'plots' directory\n")
    
    # Load Date
    try:
        print("Loading Data...")
        from tdc.single_pred import ADME
        data = ADME(name = 'Lipophilicity_AstraZeneca')
        df = data.get_data()
        print(f"✓ Loaded {len(df)} molecules\n")
    except ImportError:
        print("ERROR: TDC not installed. Install with: pip install PyTDC")
        return
    
    # Load results if available
    try:
        results_df = pd.read_csv('qsar_results.csv')
        print("✓ Loaded results from 'qsar_results.csv'\n")
    except FileNotFoundError:
        print("⚠ WARNING: 'qsar_results.csv' not found.")
        print("  Run your main QSAR script first to generate results.\n")
        results_df = None

    print("\n" + "=" * 80)
    print("CREATING VISUALIZATIONS")
    print("=" * 80)
    
    # 1. Data distribution plots
    plot_lipophilicity_distribution(df)
    plot_molecular_property_correlations(df)
    
    # 2. Model comparison plots (if results available)
    if results_df is not None:
        plot_model_comparison_bars(results_df)
        plot_metric_heatmap(results_df)
    
    # 3. Prediction quality plots (for best model)
    model, X_test, y_test, y_test_pred = plot_predicted_vs_actual(
        df, descriptor_type='Combined', model_type='XGBoost'
    )
    plot_residuals(y_test, y_test_pred, descriptor_type='Combined', model_type='XGBoost')
    
    # 4. Feature importance plots
    plot_feature_importance_rdkit(df, model_type='Random Forest')
    plot_feature_importance_rdkit(df, model_type='XGBoost', 
                                 save_path='plots/feature_importance_rdkit_xgb.png')
    
    print("\n" + "=" * 80)
    print("VISUALIZATION COMPLETE!")
    print("=" * 80)
    print("\nAll plots have been saved to the 'plots/' directory:")
    print("  1. lipophilicity_distribution.png - Shows data distribution")
    print("  2. property_correlations.png - Feature correlation heatmap")
    if results_df is not None:
        print("  3. model_comparison.png - Bar charts comparing all models")
        print("  4. metric_heatmap.png - Heatmap of all metrics")
    print("  5. predicted_vs_actual.png - Prediction quality scatter plots")
    print("  6. residual_plot.png - Residual analysis")
    print("  7. feature_importance_rdkit.png - Important features (RF)")
    print("  8. feature_importance_rdkit_xgb.png - Important features (XGB)")
    print("\n✓ All visualizations created successfully!")
    print("=" * 80)

if __name__ == "__main__":
    create_all_visualizations()