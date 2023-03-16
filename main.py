import pandas as pd
import streamlit as st
import numpy as np
from predictions import predict

st.title('Classify Trees based on height')
st.markdown('Model to classify trees into \
           Blue gum,Pine or Cypress')
st.header("Tree Features")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.text("Height")
    height_l = st.slider('Tree height(m)', 1.0, 8.0, 0.5)

with col2:
    st.text("Longest Brach")
    branch_l = st.slider('Tree branch(m)', 2.0, 4.5, 0.5)

with col3:
    st.text("Trunk Diameter")
    diameter_l = st.slider('Tree trunkt(m)', 1.0, 7.0, 0.5)

with col4:
    st.text("Leaf Thickness")
    thickness_l = st.slider('leaf thickness(m)', 0.1, 2.5, 0.5)

st.text('')

if st.button("predict type of Tree"):
   result = predict(np.array([[height_l,branch_l,diameter_l,thickness_l]]))
   st.text(result[0])














