# Matrix Decomposition Visualizer

Interactive [Streamlit](https://streamlit.io) app to enter a numeric matrix, pick a decomposition (LU, QR, Eigen, SVD, Cholesky, optional PCA), and inspect factors with reconstruction error, heatmaps, CSV downloads, and a simple timing benchmark.

## Setup

**Requirements:** Python 3.10+ recommended.

```bash
cd matrix-decomposition-app
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Open the URL shown in the terminal (typically `http://localhost:8501`).
