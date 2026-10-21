# DDR-RAM Entropy Generation & Validation Pipeline
### Automated NIST SP 800-22 & NIST SP 800-90B Randomness & Entropy Evaluation

This directory contains a platform-independent, fully parametric Python script (`ddr_entropy_gen_method.py`) to run and evaluate the **recursive prime-shift folding and interleaving pipeline** on entropy-filtered images and output formatted validation reports into Excel sheets.

---

## 1. Overview of the Shuffling Pipeline (Flowchart Structure)
The shuffling pipeline implemented in the script directly implements the 3-phase methodology:
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
The script is designed to be **platform-independent** (works on Windows, Linux, macOS) and has no dependency on heavy math packages like SciPy (uses a pure-python implementation for incomplete gamma functions).

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

## 3. Script Arguments & Parameters
Run the script using:
```bash
python ddr_entropy_gen_method.py [arguments]
```

### Parameter Table

| Flag | Parameter | Description |
| :--- | :--- | :--- |
| `-i`, `--image-dir` | **Image Directory** | Folder containing raw `.png` images to be processed. |
| `-f`, `--filter-source` | **Filter Source** | *(Optional)* Path to the 1st Filter CSV (e.g. `entropy_features_by_window.csv`) or a `.txt` file containing passing window indices. If omitted, all `.png` images in `--image-dir` are processed. |
| `-d`, `--duplicates-file`| **Duplicates List** | *(Optional)* Path to a `.txt` file containing duplicate window IDs to exclude. |
| `-s`, `--sts-path` | **NIST STS Path** | Folder containing the compiled `assess` binary and `templates/` folder. |
| `-e`, `--ea-path` | **SP 800-90B Path** | Path to the compiled `ea_non_iid` or `ea_non_iid_native` executable. |
| `-n`, `--iterations` | **Iterations** | Number of folding iterations to evaluate (e.g. `5` or `8`). |
| `-o`, `--output-excel` | **Output Excel** | Path to save the formatted result excel. If the file exists, the sheet is added/updated. |
| `--no-interleave` | **Disable Interleaving** | *(Optional)* If set, disables Phase 3 matrix interleaving and uses simple sequential concatenation. |

---

## 4. Usage Example
To run validation on RP5 1024x1024 windows for 5 iterations:
```bash
python ddr_entropy_gen_method.py \
  -i "/path/to/rp5_1024" \
  -f "/path/to/entropy_features_by_window.csv" \
  -d "/path/to/rp5_duplicates.txt" \
  -s "/path/to/sts" \
  -e "/path/to/ea_non_iid_native" \
  -n 5 \
  -o "/path/to/rp5_validation_results.xlsx"
```

---

## 5. Output Excel Formatting
The script automatically detects the image resolution (e.g., `1024x1024`) and creates a sheet named `{res}x{res}` inside the Excel file. If the file already exists, it appends/overwrites only that sheet, allowing multiple resolutions to be combined into a single file.

For each iteration, the sheet contains:
- **NIST SP 800-22 Randomness Results** (Columns A-C):
  - Pass/Fail statuses and p-values for all 17 tests.
  - Failures are highlighted in **red**.
- **NIST SP 800-90B Entropy Estimations** (Columns E-F):
  - Entropy estimates for all 10 non-IID tests.
  - Overall `H_original` value.
- Clean typography (Times New Roman, size 11) and border styles.
