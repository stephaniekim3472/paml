# NutriReceipt

NutriReceipt is a Streamlit application that estimates a grocery basket's **Purchase Health Score (PHS)** from receipt-style item lists. The app supports:

- receipt-image upload with optional OCR
- manual item entry
- pasted grocery lists
- comparison across Ridge, KNN, and MLP models
- feature-level explanations and a PHS component scorecard

## Repo Contents

- [app.py](/{PATH_TO_paml}/app.py) - Streamlit application
- [models.py](/{PATH_TO_paml}/models.py) - custom NumPy model classes used by the saved bundle
- [model_bundle.pkl](/{PATH_TO_paml}/model_bundle.pkl) - trained models and preprocessing statistics
- [Receipt_Health_Score_Predictor.ipynb](/{PATH_TO_paml}/Receipt_Health_Score_Predictor.ipynb) - training and evaluation notebook

## Local Setup

Create and activate a virtual environment:

```bash
cd {PATH_TO_paml}
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Run The Streamlit App

Start the app:

```bash
cd {PATH_TO_paml}
source .venv/bin/activate
streamlit run app.py
```

Then open the local URL Streamlit prints in the terminal, usually:

```text
http://localhost:8501
```

## Usage Modes

- `Upload Receipt` uses Anthropic OCR to extract grocery items from a receipt image.
- `Manual Input` works without any API key.
- `Paste Items` works without any API key.

If you do not configure an Anthropic key, the app still runs normally, but receipt-image OCR will stay disabled and you should use manual entry or pasted item lists instead.

## OCR Setup

Receipt OCR is optional. If you want the `Upload Receipt` tab to extract items automatically, provide an Anthropic API key before launching Streamlit.

### Option 1: Export In Your Shell

Set the key for the current terminal session, then start Streamlit:

```bash
cd {PATH_TO_paml}
source .venv/bin/activate
export ANTHROPIC_API_KEY="your_key_here"
streamlit run app.py
```

### Option 2: Store It In Local Streamlit Secrets

Create a local secrets file that is already ignored by git:

```bash
mkdir -p .streamlit
cat > .streamlit/secrets.toml <<'EOF'
ANTHROPIC_API_KEY = "your_key_here"
EOF
```

Then run:

```bash
cd {PATH_TO_paml}
source .venv/bin/activate
streamlit run app.py
```

### What To Expect

- With a valid key, the `Upload Receipt` tab will extract grocery items from JPG, PNG, JPEG, or WEBP receipt images.
- You can review and edit the extracted item list before scoring.
- If OCR is unavailable or fails, you can still switch to `Manual Input` or `Paste Items` and score the basket normally.

## Notes

- The model bundle is already included in the repo, so you should not need to retrain anything to run the demo.
- The predicted score is a **purchase-quality proxy**, not a diagnosis and not a direct measure of actual consumption.
- Never commit API keys to the repo or paste them into shared chat logs. If a key is exposed, revoke it and create a new one.
