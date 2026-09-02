# Hypoglycemia Risk Calculator — HTML Version

This is a separate repository for the HTML version of the dynamic 6-hour ICU hypoglycemia risk calculator.

## Structure

```text
hypoglycemia-risk-calculator-html/
├── app.py
├── hypoglycemia_calculator_bundle.joblib
├── requirements.txt
├── Procfile
├── .gitignore
└── templates/
    └── index.html
```

## Prediction pipeline

The model pipeline is kept consistent with the original project:

```text
11 model features
→ median imputation
→ standard scaling
→ calibrated LightGBM model
→ predict_proba()
→ 6-hour hypoglycemia risk
```

`index.html` does not contain a substitute/demo risk formula. It sends inputs to `/predict`, and the displayed probability is returned by `hypoglycemia_calculator_bundle.joblib`.

## Local run

```bash
pip install -r requirements.txt
python app.py
```

Then open `http://127.0.0.1:8000`.

## Research-use statement

This calculator is a research prototype intended for model demonstration and academic communication. It is not a validated standalone clinical decision-support product and should not replace clinical judgment.
