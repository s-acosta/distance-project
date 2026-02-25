# widefield-positioning

Per-position dF/F response extraction from widefield calcium imaging data.

Given a folder of single-frame `.tif` files, the script:
1. Loads stimulus timing from a `.mat` file in the same folder, or reconstructs it from the CSVs in `stim-generation/`
2. Computes a global baseline (F0) from the blank periods before each stimulus
3. Extracts the mean dF/F response for each screen position across all repeats
4. Saves results to `<image_dir>/output/preprocessing.npz`
5. Saves a summary figure to `<image_dir>/output/position_responses.png`

## Setup (first time on a new computer)

### 1. Get the code (sparse checkout — only downloads this folder)
```bash
git clone --filter=blob:none --sparse <repo-url>
cd distance-project
git sparse-checkout set disparity-analysis/widefield-positioning
cd disparity-analysis/widefield-positioning
```

### 2. Create a virtual environment
```bash
python -m venv .venv

# Activate — Mac / Linux:
source .venv/bin/activate

# Activate — Windows:
.venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

## Usage

```bash
# Activate the venv first (every session)
source .venv/bin/activate

# Run — only argument is the folder containing .tif frames
python analyze_positions.py /path/to/tiff/folder
```

Re-running on the same folder loads cached results from `preprocessing.npz` and skips recomputation.

## Outputs

| File | Description |
|------|-------------|
| `<image_dir>/output/preprocessing.npz` | F0 map, per-position dF/F responses, stimulus metadata |
| `<image_dir>/output/position_responses.png` | Grid of mean dF/F maps, one panel per screen position |

## Stimulus files

`stim-generation/` contains the MATLAB script and CSVs used to generate the stimulus order:
- `positions.csv` — 8 screen positions (pixels)
- `position_order.csv` — randomized presentation order across 10 repeats
- `generate_position_order.m` — MATLAB script that produced the above CSVs
