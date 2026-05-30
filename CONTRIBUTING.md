# Contributing

Contributions, issues, and feature requests are welcome!

## Setup

```bash
git clone https://github.com/<your-username>/tech-layoffs-dashboard.git
cd tech-layoffs-dashboard
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python train_models.py
streamlit run app.py
```

## Adding a New Question Intent

1. Open `app.py`
2. Add a regex pattern to the `INTENTS` dict near the top
3. Add a matching `elif intent == "your_intent":` block in the `answer()` function
4. Return `(text, fig)` — `fig` can be `None` for text-only answers

## Adding a New ML Model

1. Train it in `train_models.py` and save with `joblib.dump()`
2. Load it in the `load_models()` cache function in `app.py`
3. Add a prediction button in the sidebar section

## Code Style

- PEP 8, 4-space indents
- Docstrings on all public functions
- Keep chart logic in `utils/visualizations.py`
- Keep data encoding logic in `utils/preprocessing.py`
