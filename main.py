# Existing code in main.py
import streamlit as st
from streamlit.logger import get_logger
import pandas as pd
import seaborn as sns
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from scipy import stats
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from scipy.special import inv_boxcox
from scipy.stats import boxcox_normplot

LOGGER = get_logger(__name__)

def run():
    st.set_page_config(
        page_title="Regression",
        page_icon="📊",
    )

    st.title('Regression')

    st.subheader('Raw Data')

    # The URL of the CSV file to be read into a DataFrame
    csv_url = "https://raw.githubusercontent.com/SherryNi13/Youreka/main/infectious-diseases-by-county-year-and-sex.csv"

    # Reading the CSV data from the specified URL into a DataFrame named 'df'
    df = pd.read_csv(csv_url)

    # Display the entire dataset
    st.write(df)

    # Print the columns of the DataFrame for debugging
    st.write("Columns in the dataset:", df.columns)

    # Filter the dataset to include only rows where the disease is 'Anthrax'
    if 'Disease' in df.columns:
        df_anthrax = df[df['Disease'] == 'Anthrax']
    else:
        st.error("The column 'Disease' does not exist in the dataset.")
        return

    # Display the filtered dataset
    st.write(df_anthrax)

    # Remove duplicate rows from the dataset
    df_anthrax.drop_duplicates(keep='first', inplace=True)

    st.write('### Regression Graph')

    # Define X (features) and y (target)
    X = df_anthrax[['County']]
    y = df_anthrax['Cases']

    # Encode the 'county' column
    X['county_encode'] = LabelEncoder().fit_transform(X['County'])

    # Instantiate a linear regression model
    linear_model = LinearRegression()

    # Fit the model using the encoded 'county' and 'cases'
    linear_model.fit(X[['county_encode']], y)

    # Predict the cases using the model
    y_pred = linear_model.predict(X[['county_encode']])

    # Create a scatter plot with the regression line
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(X['county_encode'], y, color='blue', label='Actual Cases')
    ax.plot(X['county_encode'], y_pred, color='red', label='Predicted Cases')
    ax.set_xlabel('County (Encoded)')
    ax.set_ylabel('Cases')
    ax.set_title('Regression Graph: County vs Cases (Anthrax)')
    ax.legend()

    # Display the plot
    st.pyplot(fig)

    # Evaluate the accuracy of the model
    st.write('### Model Evaluation')
    st.write(f'R^2 Score: {metrics.r2_score(y, y_pred)}')
    st.write(f'Mean Absolute Error: {metrics.mean_absolute_error(y, y_pred)}')
    st.write(f'Mean Squared Error: {metrics.mean_squared_error(y, y_pred)}')
    st.write(f'Root Mean Squared Error: {np.sqrt(metrics.mean_squared_error(y, y_pred))}')

if __name__ == "__main__":
    run()
