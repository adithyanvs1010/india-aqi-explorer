# India Air Quality Explorer

A Streamlit dashboard for exploring historical city-level AQI data in India and generating next-day predictions from the included XGBoost model. The included data covers January 2015 to July 2020; predictions are not live forecasts.

## Run locally

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Push this project to a GitHub repository. Include `app.py`, `requirements.txt`, `data/`, and `models/`; do not upload `.venv/`.
2. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/) with GitHub and create an app from that repository.
3. Select the repository's branch and set the app file to `app.py`, then deploy.
4. Share the app URL to let people access the dashboard.

The app uses local files included in the repository and does not require secrets or external API credentials.
