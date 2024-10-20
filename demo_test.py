import streamlit as st
import pandas as pd

# Load your dataset
df = pd.read_csv('export.csv')

# Display the dataframe
st.write("Dataset Overview", df)

# Interactive filters or charts (optional)
selected_column = st.selectbox("Select a column to view", df.columns)
st.bar_chart(df[selected_column].value_counts())
