#!/bin/bash
# setup.sh — optional entrypoint for platforms that need it (e.g. Heroku)
# Streamlit Cloud uses requirements.txt + packages.txt directly; this is a backup.

pip install -r requirements.txt
python train_models.py
streamlit run app.py
