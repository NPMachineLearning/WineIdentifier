import streamlit as st
from model_wrapper import ModelWrapper
from nltk_text_transformer import NLPTransformer
import time

if "df" not in st.session_state:
  st.session_state.df = None

@st.cache_resource
def load_wrapper():
  wrapper = ModelWrapper("wine_identifier.pkl")
  return wrapper

wrapper = load_wrapper()

def make_prediction(text):
  data = wrapper.predict(text)
  return data

def split_frame(input_df, rows):
    df = [input_df.iloc[i : i + rows - 1, :] for i in range(0, len(input_df), rows)]
    return df

def analyse_clicked(text):
  if text == "":
    return
  st.session_state.df = make_prediction(text)

st.title("Sentiment Analysis with Wine")
st.text("App identify possible origin of wine base on your description")
desc = st.text_area(label="Description",
                    height=140,
                    placeholder="""
                    Full-bodied and complex with spicy cherry aromas
                    and flavors enhanced by a
                    savory and herbal old world style character
                    """)

st.button(label="Analyse",
          type="primary",
          on_click=analyse_clicked,
          args=(desc,))

if st.session_state.df is not None:
  tips = st.columns(1)
  with tips[0]:
    st.write("designation: The vineyard within the winery where the grapes that made the wine are from")
    st.write("variety: The type of grapes used to make the wine")

  pagination = st.container()
  bottom_menu = st.columns((4, 1, 1))

  with bottom_menu[2]:
      batch_size = st.selectbox("Page Size", options=[25, 50, 100], index=0)
  with bottom_menu[1]:
      total_pages = (
          int(len(st.session_state.df) / batch_size) if int(len(st.session_state.df) / batch_size) > 0 else 1
      )
      current_page = st.number_input(
          "Page", min_value=1, max_value=total_pages, step=1, value=1
      )
  with bottom_menu[0]:
      st.markdown(f"Page **{current_page}** of **{total_pages}** ")



  pages = split_frame(st.session_state.df, batch_size)
  pagination.dataframe(data=pages[current_page - 1], use_container_width=True)
