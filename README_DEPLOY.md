# SARIMAX Stock Forecast – Deployment (No Notebook Changes)

This repo assumes you **already ran**:
- `Pull_data.ipynb` → saved CSVs under `Stock/` and `News/`
- `stock_mlops.ipynb` → saved best models as `models/<TICKER>.pkl` and MLflow runs under `./mlruns`

## 1) (Optional) Normalize data layout
If some CSVs were saved flat as `Stock/<TICKER>.csv`, mirror them into the cap-group folders and normalize headers (no notebook edits):
```bash
python sync_data_layout.py