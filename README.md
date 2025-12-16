# AI Review Service

This project provides an improved automatic approval system for user-generated reviews. It combines a lightweight machine learning model with rule-based safeguards to deliver precise moderation decisions and explanations.

## Features

- Vietnamese-aware text preprocessing (emoji removal, repeated character normalisation, tokenisation with `underthesea`).
- TF-IDF + Logistic Regression model with configurable training parameters.
- Heuristic engine for profanity, spam-like patterns, and abnormal length detection.
- Hybrid scoring: blends model confidence with rule penalties to produce `approve`, `reject`, or `manual_review` outcomes.
- Flask inference API exposing `/review` and `/health` endpoints.
- Training script configurable via CLI, emits metrics for train/validation/test splits.

## Project Layout

```
ai-review-service/
├── app/
│   ├── __init__.py           # create_app() factory and wiring
│   ├── api.py                # Routes and Swagger definitions
│   ├── config.py             # Configuration dataclasses
│   ├── data_processing.py    # Text preprocessing helpers
│   ├── model.py              # ML training / persistence logic
│   ├── rules.py              # Rule-based checks
│   └── service.py            # Hybrid moderation orchestration
├── artifacts/                # Saved models and metadata (after training)
├── data/                     # Optional training datasets
├── requirements.txt          # Runtime dependencies
├── scripts/
│   ├── serve.py              # Launches the Flask API
│   ├── train.py              # Trains the moderation model
│   └── augment_dataset.py    # Generates synthetic multi-class training data
├── wsgi.py                   # Entry point for Flask / gunicorn
└── README.md                 # This file
```

## Getting Started

1. **Install dependencies**

   ```powershell
   pip install -r requirements.txt
   ```

2. **(Optional) Augment the dataset**

   Generate a richer multi-class corpus from `data/comment_data.csv`:

   ```powershell
   python scripts/augment_dataset.py data/comment_data.csv data/comment_data_augmented.csv --positive-count 3000 --negative-count 3000 --spam-count 3000 --neutral-sample 3000 --toxic-extra 2000
   ```

   Adjust the counts to suit your balancing needs. The command above produces ~20k rows with the five labels `positive`, `negative`, `neutral`, `spam`, `toxic`.

3. **Train the model**

   ```powershell
   python scripts/train.py data/comment_data_augmented.csv --text-column comment --label-column label
   ```

   Adjust the column names or dataset path as needed. Artifacts and metrics will appear in `artifacts/`.

4. **Run the API**

   From the project root you can now use either command:

   ```powershell
   flask run
   ```

   or

   ```powershell
   python scripts/serve.py
   ```

   Send a request:

   ```powershell
   curl -X POST http://localhost:8000/review -H "Content-Type: application/json" -d '{"comment": "This article is very helpful"}'
   ```

## Extending

- Update `DEFAULT_PROFANITY` and `DEFAULT_SUSPICION` in `app/rules.py` to tailor rule coverage.
- Override thresholds and weights through `AppConfig` if integrating into a larger application.
- Persist experiment results by enriching `scripts/train.py` with custom metric logging or MLflow integration.
