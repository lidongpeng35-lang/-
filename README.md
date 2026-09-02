# Hypoglycemia Risk Calculator — HTML Version

This is a separate repository for the HTML version of the dynamic 6-hour ICU hypoglycemia risk calculator. The original Streamlit repository remains unchanged.

## Structure

```text
├── app.py
├── requirements.txt
├── Procfile
├── .gitignore
└── templates/
    └── index.html
```

## Model source

The HTML project uses the same model bundle as the original public repository. On startup, `app.py` checks for `hypoglycemia_calculator_bundle.joblib`. If it is not present locally, the app downloads and caches the model from:

`lidongpeng35-lang/hypoglycemia-risk-calculator`

No substitute or demonstration risk formula is used.

## Prediction pipeline

```text
11 model features
→ median imputation
→ standard scaling
→ calibrated LightGBM model
→ predict_proba()
→ 6-hour hypoglycemia risk
```

The browser sends the entered values to `/predict`, and the displayed probability is returned by the Python model backend.

## Local run

```bash
pip install -r requirements.txt
python app.py
```

Then open `http://127.0.0.1:8000`.

## Research-use statement

This calculator is a research prototype intended for model demonstration and academic communication. It is not a validated standalone clinical decision-support product and should not replace clinical judgment.
