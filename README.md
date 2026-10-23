# DDR-RAM Entropy Generation & Evaluation Pipeline
### Multi-Metric Statistical Filtering & Automated NIST SP 800-22 / SP 800-90B Randomness Evaluation

This repository provides an end-to-end, platform-independent pipeline for **entropy-driven DDR-RAM region selection**, recursive prime-shift XOR folding, deterministic interleaving, and rigorous cryptographic randomness verification based on the paper:

> **"Multi-Metric Statistical Filtering for Entropy-Driven DDR-RAM Region Selection"**  
> *Oğuz Narlı, Mustafa İsmail Bayramoğlu, Meltem Kurt Pehlivanoğlu*

---

## Architecture & Workflow Overview

The framework operates in a streamlined two-step workflow:

```
[Raw DDR-RAM Window Images (.png)]
                 │
                 ▼
  STEP 1: extract_features.py ────────► Computes 7 statistical randomness metrics per window
                 │
                 ▼
     entropy_features_by_window.csv   (Input table for the 1st filter stage)
                 │
                 ▼
  STEP 2: ddr_entropy_gen_method.py ──► -f entropy_features_by_window.csv
                 │
                 ├─ 1. Single-Stage Filtering (Eq. 10)
                 ├─ 2. Prime Circular Shift & XOR Folding (M iterations)
                 ├─ 3. Deterministic Matrix Interleaving
                 └─ 4. Automated NIST SP 800-22 & NIST SP 800-90B Evaluation
                 │
                 ▼
     Formatted Excel Report (.xlsx)
```

---

## 1. Overview of the Shuffling Pipeline (Flowchart Structure)
The shuffling pipeline implemented in `ddr_entropy_gen_method.py` directly implements the 3-phase methodology:
1. **PHASE 1 – Image-Based Entropy Source**:
   - Each input grayscale PNG image is converted into an $L_0$-bit string (row-major order).
   - $N$ images are processed independently.
2. **PHASE 2 – XOR-Based Function $F(X)$, Applied $M$ Times to Every Image**:
   - For each image $i \in [1, N]$:
     - **STEP 1 (Prime Generation)**: $V = \text{int}(X[0:32], \text{base } 2)$, $p = \text{prime nearest to } V$ (tie $\to$ smaller prime).
     - **STEP 2 (Split & Circular Shift)**: Split at midpoint into $A = X[0 : |X|/2]$ and $B = X[|X|/2 : |X|]$. Rotate $B$ right by $p \bmod |B|$ times ($B' = \text{CRS}(B, p)$).
     - **STEP 3 (Bitwise XOR)**: $X' = A \oplus B'$ (bit-by-bit), halving the bit length ($|X'| = |X|/2$).
   - Repeat $M$ times to produce folded image $C_i = F^M(\text{Image}_i)$ of length $L = L_0 / 2^M$.
3. **PHASE 3 – Bit Interleaving of the Processed Images ($S_{\text{int}}$)**:
   - Construct $N \times L$ matrix $M$ whose rows are $C_1, \dots, C_N$.
   - **STEP 1**: Read the first column top-to-bottom ($C_1 \to C_N$) and append it to $S_{\text{int}}$.
   - **STEP 2**: Remove that column from the matrix.
   - **STEP 3**: Transpose the remaining matrix (rows become columns).
   - Repeat until the matrix is empty, producing the single interleaved bitstream $S_{\text{int}}$.
4. **NIST STS & SP 800-90B Randomness Tests**:
   - $S_{\text{int}}$ is passed as the input bitstream to the test suites.

---

## 2. Requirements & Setup
The scripts are designed to be **platform-independent** (works on Windows, Linux, macOS) and have no dependency on heavy math packages like SciPy (uses a pure-python implementation for incomplete gamma functions).

### Python Dependencies
Install the required packages using pip:
```bash
pip install numpy pandas pillow openpyxl
```

### NIST SP 800-22 (STS) Binary
You must have the compiled NIST Statistical Test Suite containing:
- The `assess` binary (`assess.exe` on Windows, `assess` on Linux/macOS)
- The `templates/` folder containing the template files.

To compile STS on Unix systems:
```bash
git clone --depth 1 https://github.com/terrillmoore/NIST-Statistical-Test-Suite.git sts_dir
mkdir -p sts_dir/sts/obj
make -C sts_dir/sts
# Binary will be created at sts_dir/sts/assess
```

### NIST SP 800-90B (Entropy Assessment) Binary
You must have the compiled C++ binary of the NIST SP 800-90B tool (`ea_non_iid` or `ea_non_iid_native`).

---

## 3. Step 1: Statistical Feature Extraction (`extract_features.py`)

Before applying the 1st filter, raw DDR-RAM window images are evaluated across 7 multi-metric statistical properties defined in Section IV-B (Eq. 10):

$$F(X) = (B(X) \le 0.2) \land (H(X), H_G(X) \in [0.88, 1.0]) \land (\sigma(X), \sigma_G(X) \in [50, 100]) \land (4 \le SE \le 10) \land (0.99 \le PE \le 1.0)$$

| Metric | Column Name in CSV | Filter Range (Eq. 10) | Description |
| :--- | :--- | :---: | :--- |
| **Binary Bias** | `bias` | $\le 0.20$ | Deviation from equal bit probability: $\|n_1/n_b - 0.5\|$ |
| **Shannon Entropy** | `shannon_entropy` | $[0.88, 1.0]$ | Bit-level uncertainty quantifier |
| **Block-wise Entropy**| `blockwise_entropy` | $[0.88, 1.0]$ | Average binary Shannon entropy across 16 blocks |
| **Standard Deviation**| `image_std` | $[50, 100]$ | Pixel intensity dispersion around the mean |
| **Grouped Std. Dev.** | `image_blockwise_std` | $[50, 100]$ | Average standard deviation across $4 \times 4$ blocks |
| **Spectral Entropy** | `image_spectral_entropy` | $[4, 10]$ | Normalized 2D Power Spectral Density (PSD) complexity |
| **Permutation Entropy**| `image_permutation_entropy` | $[0.99, 1.0]$ | Ordinal pattern complexity ($D=3, \tau=1$), normalized |

### Running Feature Extraction:
`extract_features.py` reads all raw `.png` images in the specified folder and computes the 7 metrics using parallel CPU multiprocessing:

```bash
python extract_features.py -i "/path/to/raw_window_images" -o "entropy_features_by_window.csv"
```

#### Arguments:
- `-i`, `--image-dir`: Directory containing input `.png` images (Required).
- `-o`, `--output-csv`: Output CSV file path (default: `entropy_features_by_window.csv`).
- `-w`, `--workers`: Number of parallel CPU worker processes (default: all available cores).

---

## 4. Step 2: Post-Processing & Evaluation Pipeline (`ddr_entropy_gen_method.py`)

Run the main pipeline script using:
```bash
python ddr_entropy_gen_method.py [arguments]
```

### Parameter Table

| Flag | Parameter | Description |
| :--- | :--- | :--- |
| `-i`, `--image-dir` | **Image Directory** | Folder containing raw `.png` images to be processed. |
| `-f`, `--filter-source` | **Filter Source** | *(Optional)* Path to the 1st Filter CSV generated by `extract_features.py` (e.g. `entropy_features_by_window.csv`) or a `.txt` file containing passing window indices. If omitted, all `.png` images in `--image-dir` are processed. |
| `-d`, `--duplicates-file`| **Duplicates List** | *(Optional)* Path to a `.txt` file containing duplicate window IDs to exclude. |
| `-s`, `--sts-path` | **NIST STS Path** | Folder containing the compiled `assess` binary and `templates/` folder. |
| `-e`, `--ea-path` | **SP 800-90B Path** | Path to the compiled `ea_non_iid` or `ea_non_iid_native` executable. |
| `-n`, `--iterations` | **Iterations** | Number of folding iterations to evaluate (e.g. `5` or `8`). |
| `-o`, `--output-excel` | **Output Excel** | Path to save the formatted result excel. If the file exists, the sheet is added/updated. |
| `--no-interleave` | **Disable Interleaving** | *(Optional)* If set, disables Phase 3 matrix interleaving and uses simple sequential concatenation. |
| `--full-dataset` | **Full Dataset Mode** | *(Optional)* Evaluates all images in the folder without sample limitation. By default, the script processes a bounded subset to enable faster evaluation and avoid long testing times. |

---

## 5. Execution Modes (Default vs. Full Dataset)

Running full NIST SP 800-22 and NIST SP 800-90B evaluation suites over very large image datasets requires significant computational time. To provide flexibility, the pipeline supports two execution modes:

1. **Default Mode (Fast Evaluation):**
   - Evaluates a bounded subset of images to enable faster results and avoid extensive test durations.
   - *Note:* Because NIST randomness tests and entropy estimators depend directly on the total input bitstream length, numerical p-values and entropy estimations obtained from this mode will naturally differ from an evaluation run over the entire dataset.

2. **Full Dataset Mode (`--full-dataset`):**
   - Evaluates all available images in the specified directory without any sample limit.
   - Intended for exhaustive statistical verification over the entire dataset.

---

## 6. End-to-End Usage Examples

### 1. Extract Statistical Features from Raw Windows:
```bash
python extract_features.py \
  -i "path/to/raw_images/zynq/1024x1024" \
  -o "zynq_1024_features.csv"
```

### 2. Run Fast Benchmark Evaluation:
```bash
python ddr_entropy_gen_method.py \
  -i "path/to/raw_images/zynq/1024x1024" \
  -f "zynq_1024_features.csv" \
  -s "tools/sts_dir/sts" \
  -e "tools/ea_dir/cpp/ea_non_iid" \
  -n 5 \
  -o "zynq_1024_fast_results.xlsx"
```

### 3. Run Exhaustive Full-Dataset Evaluation:
```bash
python ddr_entropy_gen_method.py \
  -i "path/to/raw_images/zynq/1024x1024" \
  -f "zynq_1024_features.csv" \
  -s "tools/sts_dir/sts" \
  -e "tools/ea_dir/cpp/ea_non_iid" \
  -n 5 \
  -o "zynq_1024_full_results.xlsx" \
  --full-dataset
```

---

## 7. Output Excel Formatting
The script automatically detects the image resolution (e.g., `1024x1024`) and creates a sheet named `{res}x{res}` inside the Excel file. If the file already exists, it appends/overwrites only that sheet, allowing multiple resolutions to be combined into a single file.

For each iteration, the sheet contains:
- **NIST SP 800-22 Randomness Results** (Columns A-C):
  - Pass/Fail statuses and p-values for all 17 tests.
  - Failures are highlighted in **red**.
- **NIST SP 800-90B Entropy Estimations** (Columns E-F):
  - Entropy estimates for all 10 non-IID tests.
  - Overall `H_original` value.
- Clean typography (Times New Roman, size 11) and border styles.
