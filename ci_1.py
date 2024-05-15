import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Function to generate sample data
def generate_data(num_samples, slope, intercept, noise_std):
    np.random.seed(0)
    X = 2 * np.random.rand(num_samples, 1)
    noise = noise_std * np.random.randn(num_samples, 1)
    y = intercept + slope * X + noise
    return X, y

# Function to fit linear regression model and calculate intervals
def fit_regression(X, y):
    X_with_intercept = sm.add_constant(X)
    model = sm.OLS(y, X_with_intercept).fit()
    predictions = model.predict(X_with_intercept)
    return model, predictions

# Function to calculate confidence interval
def calculate_ci(model, X, confidence_level):
    X_with_intercept = sm.add_constant(X)
    ci = model.get_prediction(X_with_intercept).conf_int(alpha=1-confidence_level)
    return ci

# Function to calculate prediction interval
def calculate_pi(model, X, confidence_level):
    X_with_intercept = sm.add_constant(X)
    pi = model.get_prediction(X_with_intercept).conf_int(obs=True, alpha=1-confidence_level)
    return pi

# Streamlit UI
st.title('CI and PI demonstration for Regression')

# User input for parameters
slope = st.sidebar.slider('Slope:', min_value=-10.0, max_value=10.0, value=3.0)
intercept = st.sidebar.slider('Intercept:', min_value=-10.0, max_value=10.0, value=4.0)
noise_std = st.sidebar.slider('Noise Standard Deviation:', min_value=0.1, max_value=5.0, value=1.0)
confidence_level = st.sidebar.slider('Confidence Level:', min_value=0.1, max_value=0.99, value=0.95, step=0.01)
num_samples = st.sidebar.slider('Number of Samples:', min_value=10, max_value=1000, value=100)

# Generate sample data
X, y = generate_data(num_samples, slope, intercept, noise_std)

# Fit linear regression model
model, predictions = fit_regression(X, y)

# Calculate confidence interval
ci = calculate_ci(model, X, confidence_level)

# Calculate prediction interval
pi = calculate_pi(model, X, confidence_level)

# Plotting
fig, ax = plt.subplots()
ax.scatter(X, y, label='Actual Data')
ax.plot(X, predictions, color='red', label='Regression Line')
ax.fill_between(X.flatten(), ci[:, 0], ci[:, 1], color='blue', alpha=0.3, label=f'{confidence_level*100}% Confidence Interval')
ax.fill_between(X.flatten(), pi[:, 0], pi[:, 1], color='green', alpha=0.3, label=f'{confidence_level*100}% Prediction Interval')
ax.set_xlabel('X')
ax.set_ylabel('y')
ax.legend()
st.pyplot(fig)
