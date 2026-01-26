# QSAR Lipophilicity Prediction - (Ongoing)

Quantitaitve Structure-Activity Relationship(QSAR) uses mathematical models predicts properties of molecules based on their chemical structure. In this program, we want to predict Lipophilicity(LogP).

- **Goal**: Predict how a molecule will behave based on its structure
- **Why?**: Testing every molecule in a lab is expensive and time-consuming
- **Solution**: Use math and computers to predict properties instead!

## What is Lipophilicity?

**Lipophilicity(LogP)** measures how much a molecules "likes" fat or water.

- Positive LogP = molecule prefet fat (lipophilic)
- Negative LogP = molecule prefers water (hydrophilic)
- It's important for drug design to have clear understanding of its affects on absorption, distribution, and toxicity.

---

### Understanding LogP Values

| LogP Value | Meaning                         | Example           |
| ---------- | ------------------------------- | ----------------- |
| < 0        | Very water-loving (hydrophilic) | Glucose (-3.2)    |
| 0-3        | Balanced                        | Aspirin (1.2)     |
| 3-5        | Fat-loving (lipophilic)         | Ibuprofen (3.5)   |
| > 5        | Very fat-loving                 | Cholesterol (7.0) |

### Real-World Example

**Aspirin (LogP = 1.2)**:

- Moderately lipophilic
- Can cross membranes to reach pain sites
- Can also be eliminated by kidneys
- Perfect balance for a pain reliever!

---

## The QSAR Pipeline

### Step-by-Step Process

```
1. Data Collection
    └─> Get molecules with known properties
        (e.g., 4000 molecules with measured LogP)

2. Descriptor Generation
    └─> Convert molecule structures into numbers
        (e.g., molecular weight, ring count, fingerprints)

3. Model Training
    └─> Machine learning finds patterns
        (With the logic of molecules with aromatic rings tend to have higher LogP)

4. Prediction
    └─> Apply the model to the new molecules
        Input: New molecule structure
        Output: Predicted LogP

5. VALIDATION
   └─> Test predictions on molecules we didn't train on
       Check if predictions match real measurements

```

---

## Why Use Machine Learning?

### Machine Learning Approach (Modern Way)

The computer finds patterns automatically:

- Can handle hundreds of features
- Captures non-linear relationships
- Adapts to complex molecular behaviors

**Example Pattern ML Might Find**:
"Molecules with 2-3 aromatic rings AND a polar group AND molecular weight 250-350 tend to have LogP around 2.5"

Patterns that would be really hard and almost impossible to be found manually!

**While:**

### Traditional Approach (Old Way)

Scientists would derive equations manually:

```
LogP ≈ (constant × molecular weight) + (constant × aromatic rings) + ...
```

**Problem**: Too simple, misses complex patterns

---

## Real-World Applications

### 1. Drug Discovery

- Screen millions of virtual molecules
- Only synthesize promising candidates
- Saves millions of dollars

### 2. Chemical Safety

- Predict environmental fate
- Will a chemical accumulate in organisms?
- Risk assessment without animal testing

### 3. Formulation Design

- Design drug delivery systems
- Predict solubility in different solvents
- Optimize cosmetics and personal care products

---

## Success Metrics:

### R² Score (Coefficient of determination):

It measures the proportion of variance in the target variable that is explained by the model. - Ranges from 0 to 1 (higher is better)

```
R² = 1 − (Sum of Squared Residuals / Total Sum of Squares)
```

**What it means**: If R² = 0.85, then 85% of the variation in LogP can be explained by our model.

### RMSE (Root Mean Square Error)

```
RMSE = sqrt((1/n) * Σ (y_i − ŷ_i)²)
```

- Average prediction error
- Same units as what you're predicting (LogP units)
- Lower is better

### MAE (Mean Absolute Error)

```
MAE = (1/n) * Σ |y_i − ŷ_i|
```

- Average absolute difference between prediction and truth
- More intuitive than RMSE
- Lower is better



## Descriptors We Use:

**Descriptors** are numbers that represent different aspects of a molecule's structure.

### 1. ECFP4 Fingerprints (Extended Connectivity Fingerprints)

ECFP4 fingerprints are a type of molecular descriptor widely used in cheminformatics for tasks like similarity search, clustering, and QSAR modeling. They represent a molecule as a fixed-length bit vector that encodes its local atomic environments.

#### What They Are

- A binary vector (list of 1s and 0s)
- Length: 2048 bits
- Each bit represents presence/absence of a specific molecular pattern

#### How They Work

**Concept**: "Circular neighborhoods"

1. Start at each atom
2. Look at atoms within 2 bonds away (radius=2, which means ECFP4)
3. Hash this pattern into a number
4. Set that bit position to 1

The “4” in ECFP4 refers to a diameter of 4 bonds (radius 2). This determines how far the algorithm explores around each atom when generating substructure identifiers.

```
ECFP4 first will make ***initial integer ID*** based on properties like:

- Atomic number

- Valence

- Number of hydrogens

- Charge

- Aromaticity

- Isotope (optional)

Then these properties are ***encoded into an integer*** using a hash function.
```

#### Why ECFP4?

- **Captures local structure**: Each part of the molecule is encoded
- **Size-independent**: Works for small and large molecules
- **Fast to compute**: No need to align or compare molecules directly
- **Similar molecules have similar fingerprints**

**Visual Example:**

```

Molecule: Ethanol (CH3-CH2-OH)

Around first carbon:
- Connected to: C, H, H, H
- Pattern: "Carbon with 3 hydrogens and 1 carbon"
- Hash to bit position 245 → Set bit 245 = 1

Around second carbon:
- Connected to: C, O, H, H
- Pattern: "Carbon with 2 hydrogens, 1 carbon, 1 oxygen"
- Hash to bit position 1523 → Set bit 1523 = 1

Around oxygen:
- Connected to: C, H
- Pattern: "Oxygen with 1 carbon and 1 hydrogen"
- Hash to bit position 892 → Set bit 892 = 1

Result: A 2048-bit vector with bits 245, 892, 1523 set to 1, others 0

```

---

### 2. MACCS Keys (Molecular ACCess System)

#### What is it?

- A fixed set of 166 structural keys
- Each key = a yes/no question about the molecule
- Standardized and interpretable

**Example: Aspirin**

```
Structure: Benzene ring with -COOH and -OCOCH3 groups

Key 23 (benzene ring): YES → 1
Key 45 (carbonyl C=O): YES → 1
Key 67 (carboxylic acid): YES → 1
Key 89 (ester group): YES → 1
... and so on for all 166 keys

Result: [1, 0, 1, 0, 0, 1, 1, 0, ...]
```

#### Why MACCS Keys?

- **Interpretable**: You know exactly what each bit means
- **Fast to compute**: Simple pattern matching
- **Proven effective**: Used in drug discovery for decades
- **Good for filtering**: "Find molecules with benzene but no nitro group"

---

### 3. RDKit Descriptors (Calculated Molecular Properties)

#### What They Are:

- Calculated numerical properties of molecules
- Continuous values (not just 0/1)
- About 200 different descriptors available in RDKit

| Descriptor            | What It Measures               | Example               |
| --------------------- | ------------------------------ | --------------------- |
| **MolWt**             | Molecular weight               | 180.16 g/mol          |
| **MolLogP**           | Calculated lipophilicity       | 1.23                  |
| **NumHDonors**        | Hydrogen bond donors           | 2 (e.g., -OH, -NH)    |
| **NumHAcceptors**     | Hydrogen bond acceptors        | 3 (e.g., =O, -O-, -N) |
| **TPSA**              | Topological polar surface area | 49.3 Ų                |
| **NumRotatableBonds** | Flexible bonds                 | 5                     |
| **NumAromaticRings**  | Aromatic rings (benzene-like)  | 2                     |
| **FractionCSP3**      | Saturated carbon fraction      | 0.33                  |
| **HeavyAtomCount**    | Non-hydrogen atoms             | 15                    |
| **RingCount**         | Total rings                    | 3                     |

#### Why Use RDKit Descriptors?

- **Physically meaningful**: Each value means something specific
- **Continuous values**: More nuanced than binary fingerprints
- **Directly related to properties**: Molecular weight affects absorption
- **Easy to interpret**: Scientists understand what they mean

#### Pros and Cons

✅ **Pros**:

- Highly interpretable
- Physically meaningful
- Related to biological/chemical properties
- Easy to visualize and understand patterns

❌ **Cons**:

- Fewer features (only ~13-200)
- May miss complex structural patterns
- Some correlation between descriptors

---

### 4. Combined Descriptors (Our Secret Weapon!)

Concatenate all descriptor types into one big feature vector:

```
Combined = [ECFP4 + MACCS + RDKit]
         = [2048 bits + 166 bits + 13 values]
         = 2227 total features
```

#### Why Combine?

Different descriptors capture different information:

- **ECFP4**: Local structural patterns
  - "This molecule has a phenyl group connected to a carbonyl"
- **MACCS**: Specific functional groups
  - "This molecule contains a carboxylic acid"
- **RDKit**: Global properties
  - "This molecule weighs 250 g/mol and has 3 rotatable bonds"

**Together**: They give a complete picture!



## Comparison Table

| Feature                | ECFP4      | MACCS     | RDKit         | Combined   |
| ---------------------- | ---------- | --------- | ------------- | ---------- |
| **Number of features** | 2048       | 166       | 13-200        | 2227+      |
| **Type**               | Binary     | Binary    | Continuous    | Mixed      |
| **Interpretability**   | Low        | High      | High          | Medium     |
| **Detail level**       | High       | Medium    | Low           | Highest    |
| **Computation time**   | Fast       | Fast      | Fast          | Fast       |
| **Best for**           | Similarity | Filtering | Understanding | Prediction |

---

## Which Descriptor Should You Use?

### General Guidelines

1. **For the best predictions**: Use **Combined**

   - Captures all aspects of molecular structure
   - Usually gives highest R² scores

2. **For understanding what matters**: Use **RDKit**

   - Can interpret feature importance
   - "Molecules with MW > 300 have higher LogP"

3. **For fast screening**: Use **MACCS**

   - Quick to compute
   - Good for filtering large libraries

4. **For similarity searching**: Use **ECFP4**
   - "Find molecules similar to this drug"
   - Standard in pharmaceutical industry

### In Our Project

We test ALL four types and compare them!

- This shows you which works best for lipophilicity
- Different properties might need different descriptors



## How Descriptors Relate to LogP

### Direct Relationships (RDKit Descriptors)

Some descriptors directly affect lipophilicity:

```
↑ Aromatic rings → ↑ LogP (rings are hydrophobic)
↑ Polar groups (OH, NH) → ↓ LogP (polar groups like water)
↑ Molecular weight → ↑ LogP (usually)
↑ Polar surface area → ↓ LogP (more polar)
```

### Pattern Relationships (ECFP4/MACCS)

Some structural patterns correlate with lipophilicity:

- Long alkyl chains → High LogP
- Multiple hydroxyl groups → Low LogP
- Benzene rings → Medium-high LogP

**Machine learning finds these patterns automatically!**



Now, we have molecular descriptors (numbers representing molecules).
We need to teach a computer to predict LogP from these numbers.

## Machine Learning for QSAR

### Our Specific Task

**Input (X)**: Molecular descriptors

```
Molecule A: [1, 0, 1, 0, ..., MolWt=180, TPSA=45, ...]
Molecule B: [1, 1, 0, 1, ..., MolWt=250, TPSA=78, ...]
```

**Output (Y)**: LogP value

```
Molecule A: LogP = 2.3
Molecule B: LogP = 1.5
```

**Goal**: Learn function f() where:

```
f(descriptors) = LogP
```



## Two Powerful Models We Use

### 1. Random Forest

### 2. XGBoost

Both are **tree-based methods**. Let's understand what that means!



## Decision Trees: The Foundation

### What is a Decision Tree?

Think of it as a flowchart of yes/no questions:

```
                    Start
                      |
              MolWt > 200?
              /           \
            Yes            No
            /               \
    NumAromaticRings > 1?   TPSA > 40?
        /        \           /        \
      Yes        No        Yes        No
      /          \         /          \
  LogP=3.5    LogP=2.1  LogP=1.2   LogP=2.8
```

**How it works**:

1. Start at top (root)
2. Answer questions about the molecule
3. Follow path to a leaf
4. Leaf gives you the prediction

### How Trees Learn

**Training process**:

1. Look at all molecules in training data
2. Find the best question to split them
   - "What question best separates high LogP from low LogP?"
3. Repeat for each branch
4. Stop when branches are "pure" (similar LogP values)

**Best split selection**:

```
Option A: Split by MolWt > 200
- Left group: LogP = [1.2, 1.5, 1.3] (variance = 0.02)
- Right group: LogP = [3.1, 2.8, 3.4] (variance = 0.08)
- Total variance = 0.10 ✓ Good split!

Option B: Split by NumRotatable > 3
- Left group: LogP = [1.2, 2.8, 1.5] (variance = 0.65)
- Right group: LogP = [3.1, 1.3, 3.4] (variance = 1.10)
- Total variance = 1.75 ✗ Bad split!

Choose Option A!
```

### Problems with Single Trees

❌ **Overfitting**: Tree becomes too specific to training data

```
If molecule has MolWt=180.16 AND TPSA=63.6 AND NumRings=1 AND ...
Then LogP = 1.19

(This only works for aspirin!)
```

❌ **Unstable**: Small changes in data → completely different tree

**Solution**: Use MANY trees (ensemble methods)!

---

### How Random Forest Works

```

                    Training Data (1000 molecules)
                            |
        ┌───────────────────┼───────────────────┐
        ↓                   ↓                   ↓
    Tree 1              Tree 2              Tree 100
(random sample)     (random sample)     (random sample)
  predicts 2.3        predicts 2.1        predicts 2.4
        \                   |                   /
         \                  |                  /
          \                 |                 /
           \                ↓                /
            \          Average all          /
             \         predictions         /
              \____________ | _____________/
                           ↓
                  Final prediction: 2.27

```

### The "Random" Part

**Two sources of randomness**:

1. **Random Sampling**(Bootstrap)

- Each tree sees a random 60-70% of training data
- Some molecules appear multiple times, some not at all

2. **Random Features**

- At each split, only consider random subset of features
- Prevents trees from being too similar
- Forces trees to find different patterns

### Example: Training Random Forest

```
Original data: 1000 molecules

Tree 1:
- Randomly sample 700 molecules (with replacement)
- At each split, consider random 50 features out of 2227
- Learns pattern: "High aromatic ring count → High LogP"

Tree 2:
- Randomly sample 700 molecules (different ones!)
- At each split, consider random 50 features (different ones!)
- Learns pattern: "Low TPSA → High LogP"

Tree 3:
- Another random sample...
- Learns pattern: "High MolWt + Low polar groups → High LogP"

... 97 more trees ...

Each tree learns slightly different patterns!
```

### Why Random Forest is Powerful

✅ **Reduces overfitting**: Each tree overfits differently, averaging cancels it out

✅ **Stable**: Robust to outliers and noise

✅ **Feature importance**: Can see which descriptors matter most

```

Feature importance:
1. NumAromaticRings: 18%
2. TPSA: 15%
3. MolWt: 12%
4. (etc.)
```

✅ **No data preprocessing needed**: Works with any scale of features

✅ **Handles non-linear relationships**: No need to assume linear relationships

### Hyperparameters (Settings)

```python

RandomForestRegressor(
    n_estimators=100,# Number of trees
    max_depth=20,# Maximum depth of each tree
    min_samples_split=5,# Minimum samples to split a node
    random_state=42# For reproducibility
)

```

**Tuning tips**:

- More trees (n_estimators) → Better but slower
- Deeper trees (max_depth) → More complex patterns but risk overfitting
- Larger min_samples_split → Simpler trees, less overfitting

### How XGBoost Works

```
Initial prediction: Average LogP = 2.0 (for all molecules)

Tree 1:
- Looks at errors: "Which molecules did we predict badly?"
- Builds tree to predict these errors
- New prediction = 2.0 + 0.3 = 2.3

Tree 2:
- Looks at remaining errors after Tree 1
- Builds tree to fix these
- New prediction = 2.3 + 0.15 = 2.45

Tree 3:
- Looks at remaining errors after Tree 1 + Tree 2
- Builds tree to fix these
- New prediction = 2.45 + 0.08 = 2.53

... continue for 100 trees ...

Final prediction = sum of all tree predictions
```

### Visual Example

```
True LogP: 3.5

Initial: Predict 2.0 (error = +1.5)
After Tree 1: Predict 2.3 (error = +1.2) ← Fixed some error
After Tree 2: Predict 2.6 (error = +0.9) ← Fixed more
After Tree 3: Predict 2.9 (error = +0.6) ← Getting closer
After Tree 4: Predict 3.2 (error = +0.3)
After Tree 5: Predict 3.4 (error = +0.1) ← Very close!
```

### Gradient Boosting Explained

**"Gradient"** = Direction of steepest improvement

Think of it as hiking down a mountain:

1.  You're at high error (top of mountain)
2.  Look around: Which direction reduces error most?
3.  Take a step in that direction
4.  Repeat until error is minimized (bottom of valley)

    Each tree = one step down the mountain

### Hyperparameters (Settings)

```python
XGBRegressor(
n_estimators=100,               # Number of trees
    max_depth=6,                # Maximum depth (usually shallower than RF)
    learning_rate=0.1,          # Step size (important!)
    subsample=0.8,              # Fraction of samples for each tree
    colsample_bytree=0.8        # Fraction of features for each tree
)
```

**Key parameter: learning_rate**

- Controls how much each tree contributes
- Small learning_rate (0.01) + Many trees → Better but slower
- Large learning_rate (0.3) + Fewer trees → Faster but risk overfitting

```
With learning_rate=0.1:
Tree 1 adds: 0.1 × (its prediction)
Tree 2 adds: 0.1 × (its prediction)
Slowly approaches the answer

With learning_rate=0.5:
Tree 1 adds: 0.5 × (its prediction)
Tree 2 adds: 0.5 × (its prediction)
Faster but might overshoot
```



## Random Forest vs XGBoost: Head-to-Head

| Aspect                         | Random Forest                  | XGBoost                        |
| ------------------------------ | ------------------------------ | ------------------------------ |
| **How trees are built**        | Independently (parallel)       | Sequentially (boosting)        |
| **Training speed**             | Fast                           | Slower                         |
| **Prediction speed**           | Fast                           | Fast                           |
| **Accuracy**                   | Very good                      | Excellent                      |
| **Overfitting risk**           | Low                            | Medium (needs tuning)          |
| **Hyperparameter sensitivity** | Low                            | High                           |
| **When to use**                | Quick baseline, stable results | Best performance, competitions |

### Which Should You Use?

**Use Random Forest when**:

- You want quick results
- You're a beginner
- Your dataset is small-medium
- You want stable, reliable performance

**Use XGBoost when**:

- You need the absolute best accuracy
- You have time to tune hyperparameters
- Your dataset is large
- You're in a competition

**Best practice**: Try both! (That's what we do in our code!)



## Training and Testing: The Critical Split

### Why Split Data?

**Problem**: If we test on training data, we just memorize!

```
Bad approach:
1. Train on 1000 molecules
2. Test on same 1000 molecules
3. Get 99% accuracy
4. Apply to new molecule → Terrible prediction!
(The model just memorized, didn't learn patterns)
```

**Solution**: Hold out some data for testing

```
Good approach:
1. Split: 800 training + 200 testing
2. Train on 800 molecules only
3. Test on the separate 200 molecules
4. If performance is good on test set → Model learned real patterns!
```

### The 80/20 Rule

```
Total data: 1000 molecules
Training set (80%): 800 molecules
- Used to build the model
- Model sees these during training

Test set (20%): 200 molecules
- Never used during training
- Used only to evaluate final performance
- Simulates "new molecules we haven't seen"
```

### Example

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train on training set
model.fit(X_train, y_train)

# Predict on test set (never seen before!)
predictions = model.predict(X_test)

# Evaluate print
print( f"Test R² Score: { r2_score ( y_test , predictions ) } " )
```





## How Models Make Predictions: Step-by-Step

### Random Forest Prediction

```
New molecule: Caffeine
Descriptors: [1, 0, 1, 1, ..., MolWt=194, TPSA=58, NumRings=2, ...]

Tree 1: Predicts LogP = -0.2
Tree 2: Predicts LogP = -0.1
Tree 3: Predicts LogP = 0.1
Tree 4: Predicts LogP = -0.3
... (96 more trees)
Tree 100: Predicts LogP = 0.0

Final prediction = Average = -0.07
(Actual caffeine LogP = -0.07, perfect!)
```

### XGBoost Prediction

```

New molecule: Caffeine
Base prediction: 2.0 (average LogP of training set)
Tree 1: Says "subtract 1.8" → 2.0 - 1.8 = 0.2
Tree 2: Says "subtract 0.15" → 0.2 - 0.15 = 0.05
Tree 3: Says "subtract 0.08" → 0.05 - 0.08 = -0.03
Tree 4: Says "add 0.02" → -0.03 + 0.02 = -0.01
... (96 more trees making tiny adjustments)
Tree 100: Says "subtract 0.01" → -0.06

Final prediction = -0.06
(Actual caffeine LogP = -0.07, very close!)
```

## Interpreting Results

### Understanding Metrics

**R² Score = 0.85**

- 85% of LogP variation is explained by our model
- Very good! (0.7+ is good for QSAR)

**RMSE = 0.5**

- On average, predictions are ±0.5 LogP units off
- For LogP range of -3 to +7, this is quite accurate

**MAE = 0.4**

- Average error is 0.4 LogP units
- Most predictions within ±0.5 of true value

### What Can Go Wrong?

**1. Overfitting**

```
Training R² = 0.99 (amazing!)
Test R² = 0.45 (terrible!)
Problem: Model memorized training data, can't generalize
Solution: Simpler model, more training data, or better regularization
```

**2. Underfitting**

```
Training R² = 0.50
Test R² = 0.48
Problem: Model too simple, missing important patterns
Solution: More complex model, more features, or better descriptors
```

**3. Just Right!**

```
Training R² = 0.88
Test R² = 0.85
Good! Model learned real patterns and generalizes well
```



## Practical Tips

### 1. Always Check Both Models

- Random Forest might work better for some descriptor types
- XGBoost might excel with others
- Try both!

### 2. Look at Predictions vs Truth

```
Ideal:
Predicted LogP = True LogP
(Points fall on diagonal line)
Good model:
Points cluster near diagonal
Bad model:
Points scattered everywhere
```

### 3. Feature Importance

```python
importance = model . feature_importances_
print ( "Top 5 important features:" )
for idx in importance . argsort ( ) [ - 5 : ] [ : : - 1 ] :
  print ( f" { feature_names [ idx ] } : { importance [ idx ] : .3f } " )
```

### 4. Cross-Validation (Advanced)

Instead of one 80/20 split, do multiple splits and average results.
More reliable estimate of true performance!



## What is RDKit?
**RDKit** = **R**eusable **D**ata **Kit** for Cheminformatics

### Definition
RDKit is a collection of tools for working with chemical structures using Python.

### What Can RDKit Do?

1. **Read and write molecular structures**
- SMILES, MOL files, SDF files, etc.


2. **Calculate molecular properties**
- Molecular weight, LogP, number of atoms, etc.


3. **Generate molecular fingerprints**
- ECFP, MACCS keys, etc.


4. **Substructure searching**
- "Find all molecules with a benzene ring"


5. **Molecular visualization**
- Draw 2D structures


6. **3D structure generation**
- For docking and conformer analysis

---

## RDKit Basics: Core Concepts 
### 1. The Mol Object 

**Everything in RDKit starts with a Mol object** 

``` python 
from rdkit import Chem                           # Create a Mol object 
from SMILES mol = Chem . MolFromSmiles ( 'CCO' ) # Ethanol 
# What is mol? 
# - It's a Python object representing the molecule 
# - Contains atoms, bonds, and their connections 
# - The foundation for all RDKit operations 
```

### 2. Reading Smiles
```python
from rdkit import Chem

# Convert SMILES to Mol
smiles = 'CCO'
mol = Chem.MolFromSmiles(smiles)

# Check if valid
if mol is None:
    print("Invalid SMILES!")
else:
    print("Valid molecule!")
    
# Get back SMILES (canonical form)
canonical_smiles = Chem.MolToSmiles(mol)
print(canonical_smiles)  # 'CCO'
```
---

### Working with molecular Descriptors
**Calculating Basic Properties**

```python
from rdkit import Chem
from rdkit.Chem import Descriptors

# Create molecule
mol = Chem.MolFromSmiles('CC(=O)O')  # Acetic acid

# Calculate properties
mol_weight = Descriptors.MolWt(mol)
logp = Descriptors.MolLogP(mol)
h_donors = Descriptors.NumHDonors(mol)
h_acceptors = Descriptors.NumHAcceptors(mol)
tpsa = Descriptors.TPSA(mol)

print(f"Molecular Weight: {mol_weight:.2f}")
print(f"LogP: {logp:.2f}")
print(f"H-Bond Donors: {h_donors}")
print(f"H-Bond Acceptors: {h_acceptors}")
print(f"Polar Surface Area: {tpsa:.2f}")

# Output:
# Molecular Weight: 60.05
# LogP: -0.17
# H-Bond Donors: 1
# H-Bond Acceptors: 2
# Polar Surface Area: 37.30
```

### Common Descriptors Explained
| Descriptor | Meaning | Why It Matters | 
|------------|---------|----------------| 
| **MolWt** | Molecular weight | Affects absorption | 
| **MolLogP** | Lipophilicity | Predicts membrane permeability | 
| **NumHDonors** | H-bond donors (-OH, -NH) | Affects solubility | 
| **NumHAcceptors** | H-bond acceptors (=O, -N) | Affects binding | 
| **TPSA** | Topological polar surface area | Blood-brain barrier penetration | 
| **NumRotatableBonds** | Flexible bonds | Molecular flexibility | 
| **NumAromaticRings** | Aromatic rings | Structural feature | 
| **FractionCSP3** | Saturated carbons | 3D vs flat shape | 
| **HeavyAtomCount** | Non-hydrogen atoms | Molecular size |
 ---

