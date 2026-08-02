# Next Word Predictor

A polished Streamlit app that predicts the next word in a sequence using a trained LSTM model.

## Project files

- `app.py` - Streamlit application
- `lstm_model.h5` - trained LSTM model
- `tokenizer.pkl` - tokenizer used during training
- `max_len.pkl` - sequence length metadata
- `requirements.txt` - Python dependencies for deployment
- `.streamlit/config.toml` - Streamlit runtime and theme config
- `Procfile` - startup command for compatible hosts

## Local run

### Using your virtual environment

```bash
.venv\Scripts\streamlit run app.py
```

### Using Python directly

```bash
python -m streamlit run app.py
```

## Deploy to Streamlit Community Cloud

1. Push this full folder to a GitHub repository
2. Make sure these files are committed:
   - `app.py`
   - `lstm_model.h5`
   - `tokenizer.pkl`
   - `max_len.pkl`
   - `requirements.txt`
   - `.streamlit/config.toml`
3. Go to Streamlit Community Cloud
4. Create a new app from your GitHub repo
5. Set the main file path to `app.py`
6. Deploy

## Deploy to Render

1. Create a new Web Service from the GitHub repo
2. Runtime: Python
3. Build command:

```bash
pip install -r requirements.txt
```

4. Start command:

```bash
streamlit run app.py --server.port $PORT --server.address 0.0.0.0
```

## Deploy to Hugging Face Spaces

1. Create a new Space
2. Select **Streamlit** SDK
3. Upload the full project files
4. Ensure `requirements.txt` is present
5. Set `app.py` as the app entry point if prompted

## Common deployment issues fixed in this project

- Stable asset loading using paths relative to `app.py`
- Explicit dependency list in `requirements.txt`
- Headless Streamlit config for cloud environments
- Port binding support through the Procfile command
- Better runtime error messages for missing model files

## Notes

- The model file is large, so deployment may take longer than simple apps
- TensorFlow on free tiers can have cold starts
- If a platform has memory limits, Streamlit Community Cloud or Render is usually a better fit than lighter hosts
