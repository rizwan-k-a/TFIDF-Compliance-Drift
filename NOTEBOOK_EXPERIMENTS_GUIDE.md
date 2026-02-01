# experiments.ipynb - Complete Guide

## Overview

The `experiments.ipynb` notebook provides comprehensive hyperparameter sensitivity analysis and validation for the TF-IDF compliance drift system. It demonstrates:
- How hyperparameters affect feature extraction
- Differences between TF and IDF variants
- Validation of manual vs sklearn implementations

---

## Notebook Structure

### Cell 1: Setup (59 lines) ✅
**Purpose**: Initialize environment and load data

**What it does:**
- Imports pandas, numpy, sklearn, matplotlib, seaborn
- Loads 10 documents from `data/internal/` directory
- Defines `preprocess_text()` locally for flexibility
- Removes stopwords and special characters
- Cleans all documents for analysis

**Output:**
```
✓ Loaded N documents
✓ Preprocessed N documents
```

**Key Functions:**
- `preprocess_text(text)`: Local preprocessing function
  - Lowercase conversion
  - Special character removal
  - Stopword filtering
  - Whitespace normalization

---

### Cell 2: Baseline TF-IDF Analysis (29 lines) ✅
**Purpose**: Establish baseline metrics with default parameters

**What it does:**
- Creates TfidfVectorizer with default parameters
- Fits on sample documents
- Calculates sparsity and density
- Identifies top 10 features by TF-IDF sum

**Output Example:**
```
Matrix shape: 10 documents × N features
Non-zero elements: M
Sparsity: XX.XX%
Matrix density: XX.XX%

Top 10 Features by TF-IDF Sum:
  1. compliance              → X.XXXX
  2. procedure               → X.XXXX
  ...
```

**Key Metrics:**
- **Matrix shape**: (n_docs, n_features)
- **Non-zero elements**: Count of non-zero TF-IDF values
- **Sparsity**: Percentage of zero values
- **Density**: Percentage of non-zero values

---

### Cell 3: Hyperparameter Sensitivity Sweep (36 lines) ✅
**Purpose**: Test parameter combinations and capture metrics

**What it does:**
- Tests 16 combinations (4 min_df × 4 max_df values)
- **min_df values**: [0.01, 0.05, 0.1, 0.2]
- **max_df values**: [0.8, 0.9, 0.95, 1.0]
- For each combination:
  - Creates TfidfVectorizer
  - Records: min_df, max_df, n_features, nnz, sparsity
  - Stores results in DataFrame

**Output Example:**
```
Running hyperparameter sweep...
Testing 4 × 4 = 16 combinations

✓ min_df=0.01, max_df=0.80 → 1234 features, sparsity=95.23%
✓ min_df=0.01, max_df=0.90 → 1567 features, sparsity=95.67%
...

Summary Statistics:
              n_features        sparsity
            min  mean  max    min  mean  max
min_df
0.01         ...   ...  ...    ...  ...  ...
```

**Results Stored:** `df_sweep` DataFrame with all 16 results

---

### Cell 4: Hyperparameter Visualization (35 lines) ✅
**Purpose**: Visualize the impact of hyperparameters

**What it produces:**
- **Plot 1**: Feature count vs min_df (lines for each max_df)
  - X-axis: min_df (log scale)
  - Y-axis: Number of features
  - Multiple lines (one per max_df value)
  - Shows how min_df affects feature count

- **Plot 2**: Sparsity vs min_df (lines for each max_df)
  - X-axis: min_df (log scale)
  - Y-axis: Sparsity (%)
  - Multiple lines (one per max_df value)
  - Shows how min_df affects sparsity

**Output File:**
```
✓ Saved visualization to ../results/hyperparameter_sweep.png
```

**Key Insights:**
- Higher min_df → fewer features
- Higher max_df → more features
- Tradeoff between coverage and noise reduction

---

### Cell 5: TF Variant Comparison (63 lines) ✅
**Purpose**: Compare all 5 TF variants for word "compliance"

**TF Variants Tested:**
1. **Raw count**: Simple term frequency
   - Formula: count
   
2. **Log-normalized**: 1 + log(count)
   - Formula: 1 + log(count) if count > 0 else 0
   - Reduces impact of repeated terms
   
3. **Double-normalized**: 0.5 + 0.5 × (count / max_count)
   - Formula: 0.5 + 0.5 × (count / max)
   - Scales between 0.5 and 1.0
   
4. **Augmented**: count / max_count
   - Formula: count / max
   - Normalized by maximum term count in document
   
5. **Boolean**: 1 if present, 0 otherwise
   - Formula: 1 if count > 0 else 0
   - Binary presence/absence

**Output:**
- Table comparing 5 TF variants across 5 sample documents
- Bar chart visualization comparing variants

**Example Table:**
```
Document  Raw Count  Log-normalized  Double-normalized  Augmented  Boolean
Doc 1          5         1.609               0.787         1.000     1.0
Doc 2          0         0.000               0.500         0.000     0.0
...
```

**Key Insights:**
- Different variants have different sensitivity ranges
- Log-normalized reduces impact of frequent terms
- Double-normalized maintains bounded range

---

### Cell 6: IDF Variant Comparison (85 lines) ✅
**Purpose**: Compare all 4 IDF variants for word "compliance"

**IDF Variants Tested:**
1. **Standard IDF**: log(N / df)
   - N = total documents
   - df = documents containing term
   - Simple and interpretable
   
2. **Smooth IDF** (sklearn default): log((N + 1) / (df + 1)) + 1
   - Adds 1 to prevent division by zero
   - Smoother distribution
   - Standard in production systems
   
3. **Max IDF**: log(max_df / df)
   - Uses maximum possible document frequency
   - Normalizes differently
   
4. **Probabilistic IDF**: log((N - df) / df)
   - Emphasizes rarity through (N - df) numerator
   - Different weighting philosophy

**Output:**
- Corpus statistics (N, df)
- Table comparing 4 IDF variants
- Bar chart with value labels
- Interpretation of differences

**Example Output:**
```
Total documents (N): 10
Documents containing 'compliance': 7

IDF Variants Comparison Table:
IDF Variant                          Value
Standard: log(N/df)                  0.3567
Smooth: log((N+1)/(df+1)) + 1        1.3365
Max IDF: log(max_df/df)              0.3567
Probabilistic: log((N-df)/df)       -0.8473
```

**Key Insights:**
- Standard and Max IDF often similar
- Smooth IDF larger due to +1 constant
- Probabilistic can be negative (log of fraction < 1)

---

### Cell 7: Manual vs Sklearn TF-IDF Validation (125 lines) ✅
**Purpose**: Validate manual implementation against sklearn

**What it does:**
- Computes TF-IDF manually for 3 sample words
- Computes same words with sklearn TfidfVectorizer
- Compares values with 1% tolerance
- Reports match percentage

**Sample Words:** "compliance", "procedure", "policy"
**Test Corpus:** First 5 preprocessed documents

**Computation Steps:**
1. Manual TF-IDF:
   - TF: log-normalized (1 + log(count))
   - IDF: smooth ((N+1)/(df+1)) + 1
   - TF-IDF: TF × IDF
   - L2 normalize final values

2. Sklearn TF-IDF:
   - Extract values for same words
   - Already L2 normalized

**Output:**
- Document-by-word comparison table
- Individual differences (tolerance check)
- Summary statistics
- Match percentage and validation result

**Example Output:**
```
Word: 'compliance'
  Document | Manual TF-IDF | Sklearn TF-IDF | Diff      | Match
  ──────────────────────────────────────────────────────────
  1        |      0.123456 |      0.123789 | 0.000333  | ✓
  2        |      0.054321 |      0.054567 | 0.000246  | ✓
  ...

VALIDATION SUMMARY:
Total comparisons: 15
Matches (within 1%): 15
Match percentage: 100.0%

✅ VALIDATION PASSED: 100.0% match (threshold: 95%)
```

**Validation Criteria:**
- ✅ PASS: ≥95% match
- ⚠️ WARNING: <95% match

---

## Running the Notebook

### Prerequisites
```bash
pip install -r requirements-dev.txt
```

### Quick Start
```bash
jupyter notebook notebooks/experiments.ipynb
```

### Run All Cells
```bash
# From Jupyter UI: Kernel → Restart & Run All
# Or from command line:
jupyter nbconvert --to notebook --execute --output experiments.ipynb notebooks/experiments.ipynb
```

### Run Individual Cells
Click on cell and press `Shift + Enter` to execute

---

## Expected Execution Results

### Cell 1: Setup ✅
- Loads 10 documents
- Preprocesses them
- Ready for analysis

### Cell 2: Baseline ✅
- Shows default vectorizer stats
- Typically ~500-2000 features depending on corpus
- Sparsity usually 95-98%

### Cell 3: Sweep ✅
- Completes all 16 combinations
- All should succeed
- Creates df_sweep DataFrame

### Cell 4: Visualization ✅
- Produces 2x1 subplot figure
- Saves PNG to results/
- Shows clear trends

### Cell 5: TF Comparison ✅
- Shows 5 variants for "compliance"
- Produces comparison table and bar chart
- All 5 variants should have different ranges

### Cell 6: IDF Comparison ✅
- Shows 4 variants for "compliance"
- Produces comparison table and bar chart
- Displays interpretation text

### Cell 7: Validation ✅
- Compares 15 values (3 words × 5 docs)
- Should show ✅ VALIDATION PASSED
- Match percentage ≥95%

---

## Common Issues & Fixes

### Issue: Module import errors
**Solution:** Ensure `src/` is in Python path (Cell 1 does this automatically)

### Issue: FileNotFoundError for data files
**Solution:** Run from `notebooks/` directory, or adjust path in Cell 1

### Issue: Visualization not showing
**Solution:** Ensure matplotlib backend is enabled, try `%matplotlib inline` in cell 0

### Issue: Validation fails (<95% match)
**Solution:** Check that both sklearn and manual implementations use smooth IDF

### Issue: Sweep takes too long
**Solution:** Reduce corpus size in Cell 1 (change `[:10]` to `[:5]`)

---

## Key Takeaways

### Hyperparameter Tuning
- **min_df**: Filters rare terms (lower = more features)
- **max_df**: Filters common terms (higher = more features)
- Tradeoff: More features capture more info but increase sparsity

### TF Variants
- **Log-normalized**: Reduces impact of frequency differences
- **Double-normalized**: Maintains bounded range (0.5-1.0)
- **Boolean**: Only captures presence/absence

### IDF Variants
- **Standard**: Simple, fundamental formula
- **Smooth**: Production-ready, avoids zeros
- **Probabilistic**: Alternative weighting scheme

### Validation
- Manual and sklearn implementations should match within 1%
- Small differences due to floating-point precision
- L2 normalization ensures comparable values

---

## Next Steps

1. **Tune Parameters**: Modify min_df/max_df in Cell 3 based on use case
2. **Add Custom Words**: Change sample words in Cells 5-6
3. **Extend Validation**: Add more documents or words to Cell 7
4. **Experiment Locally**: Try different preprocessing in Cell 1
5. **Benchmark Performance**: Add timing measurements

---

## File References

- **Notebook**: `notebooks/experiments.ipynb`
- **Data**: `data/internal/*.txt`
- **Output**: `results/hyperparameter_sweep.png`
- **Source Code**: `src/manual_tfidf_math.py`, `src/preprocess.py`

---

## Documentation References

- [TF-IDF Theory](IMPLEMENTATION_SUMMARY.md#tfidf-math)
- [Hyperparameter Tuning](IMPLEMENTATION_SUMMARY.md#hyperparameter-guide)
- [Testing Framework](QUICK_START_TESTING.sh)

---

**Last Updated**: February 1, 2026
**Status**: ✅ Complete and Ready to Run
**Cells**: 14 (7 code + 7 markdown)
**Lines of Code**: 360+

