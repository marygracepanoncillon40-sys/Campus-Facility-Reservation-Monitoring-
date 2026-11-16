# Campus Facility Reservation Monitoring (Section A)

Streamlit deployment of the midterm data-mining model: predicts **cancellation / no-show** of facility
reservations and supports Facility Administrator review. Predictions are decision support only.

## Modules
| Page | File | Purpose |
|---|---|---|
| Dataset | `modules/data_processing.py` | CSV upload, schema check, preview, null/duplicate summary |
| Data Analysis | `modules/exploratory_analysis.py` | Stats, class distribution, 4 interactive Plotly charts |
| Prediction | `modules/prediction.py` | Cached saved pipeline, single/batch prediction, metrics, consistency check |
| Dashboard | `modules/dashboard.py` | KPIs, review queue, interpretation, follow-up, decision log |
| System Design | `modules/system_design.py` | FR/NFR, actors, use case diagram, process flow |

## Run locally
    pip install -r requirements.txt
    python scripts/generate_data.py   # synthetic data (skip if you use your midterm data)
    python scripts/train_model.py     # saves models/trained_model.joblib (do once, not per page load)
    python scripts/make_diagrams.py
    streamlit run app.py

## Using your own midterm work
1. Put your dataset at `data/sample_reservations.csv` (authorized/anonymized only).
2. Edit `TARGET`, `CATEGORICAL`, `NUMERIC`, `LEAKAGE` and the estimator in `scripts/train_model.py` to match your notebook, then rerun it.
3. Explain differences from the original midterm problem in your deployment report.

## Deploy
Push to GitHub → https://share.streamlit.io → New app → select repo/branch → entry point `app.py`.
Keep `scikit-learn` pinned to the version that trained the model (see `models/model_metadata.json`).

## Limitations
Bundled data and model are synthetic. Probabilities are not guaranteed outcomes. Decision log is session-only.
