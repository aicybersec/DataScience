# Open uber.py in your favorite IDE or text editor, then add these lines:

import streamlit as st
import pandas as pd
import numpy as np

# Apps title

st.title('Uber pickups in Melbourne')

DATE_COLUMN = 'date/time'
DATA_URL = ('https://s3-us-west-2.amazonaws.com/'
            'streamlit-demo-data/uber-raw-data-sep14.csv.gz')
####################################################################
run Streamlit from the command line:

streamlit run uber.py
####################################################################

@st.cache

def load_data(nrows):
    data = pd.read_csv(DATA_URL, nrows=nrows)
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)
    data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
    return data

data_load_state = st.text('Loading data...')
data = load_data(10000)
####################################################################
Replace the line data_load_state.text('Loading data...done!') with this:

data_load_state.text("Done! (using st.cache)")
####################################################################

# Inspect the raw data

st.subheader('Raw data')
st.write(data)

# Draw a histogram

st.subheader('Number of pickups by hour')

hist_values = np.histogram(
    data[DATE_COLUMN].dt.hour, bins=24, range=(0,24))[0]

st.bar_chart(hist_values)

# Plot data on a map

st.subheader('Map of all pickups')

st.map(data)

st.subheader('Map of all pickups')
st.map(data)

# Filter results with a slider

hour_to_filter = st.slider('hour', 0, 23, 17)  # min: 0h, max: 23h, default: 17h
filtered_data = data[data[DATE_COLUMN].dt.hour == hour_to_filter]
st.subheader(f'Map of all pickups at {hour_to_filter}:00')
st.map(filtered_data)

# Use a button to toggle data

st.subheader('Raw data')
st.write(data)

if st.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.write(data)