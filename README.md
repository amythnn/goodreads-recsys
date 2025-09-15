# Goodreads Recsys 📚✨

## Project Overview
**Goodreads Recsys** is an end-to-end recommendation system pipeline built on collaborative filtering (CF). It demonstrates the full workflow on Goodreads-style ratings data: data cleaning, quick EDA, training UserKNN & ItemKNN models (via `surprise`), evaluation with RMSE and Precision/Recall@K, and exporting example top-N recommendations. The repo is organized for clarity and reproducibility (comments, CLI flags, artifacts on disk, and a report scaffold).

---

## Installation

### 1. Clone the repository
git clone https://github.com/amythnn/goodreads-recsys.git
cd goodreads-recsys

### 2. (Optional) Create a virtual environment
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate

### 3. Install dependencies
pip install -r requirements.txt

---

## Usage
Run the pipeline on your dataset (replace with your local paths):

python Code/goodreads_pipeline.py \
  --books_csv "/path/to/books.csv" \
  --ratings_csv "/path/to/ratings.csv" \
  --out_dir artifacts

Outputs will be saved in the `artifacts/` folder:
- eda_summary.json — rating distribution stats
- metrics.json — RMSE + precision/recall
- sample_recs.json — example top-N recommendations

---

## Repo Structure
code/         # pipeline code (goodreads_pipeline.py)
reports/      # markdown reports & evaluation writeups
artifacts/    # generated outputs (ignored in git)
data/         # local datasets (ignored in git)
README.md
requirements.txt
.gitignore
LICENSE
