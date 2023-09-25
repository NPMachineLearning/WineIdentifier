## Streamlit app

Streamlit app for sentiment analysis with wine

### Deployment

- Clone this repo
- Use anaconda to create a new environment from this repo
  `conda env create -f env.yml -p env`
- use pip to export **requirements.txt** file
  `pip freeze > requirements.txt`
- Use Git push to your repo
- Follow Streamlit deployment tutorial [here](https://docs.streamlit.io/streamlit-community-cloud/deploy-your-app#deploy-your-app-1)

### Files

- **app.py**: Streamlit app
- **model_wrapper.py**: A wrapper for sklearn model
- **nltk_text_transformer.py**: Sklearn custom transformer to transform text data
- **wine_identifier.pkl**: Trained model which is used to make prediction.
- **env.yml**: Anaconda environment file
- **requirements.txt**: Libraries requirement file for Streamlit deployment
