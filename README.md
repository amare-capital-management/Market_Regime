
<img width="780" height="253" alt="ACM w color" src="https://github.com/user-attachments/assets/3a596e9e-9337-4ceb-b1e7-c1a9ca3a7012" />

### The 8.Market_Regime.py script is designed to analyze market regimes and implement a pairs trading strategy. Here's a detailed breakdown of its functionality:

*1. Imports and Setup:*

Utilizes libraries like pandas, numpy, yfinance, statsmodels, and matplotlib for data analysis, statistical modeling, and visualization.
Sets up directories for saving results and plots.

*2. Helper Functions:*

adf_test: Performs the Augmented Dickey-Fuller (ADF) test to check if a time series is stationary.
compute_hurst: Computes the rolling Hurst exponent to classify market regimes (mean-reverting or trend-following).
pairs_trading_ratio: Implements a pairs trading strategy for two assets:
Calculates the hedge ratio using Ordinary Least Squares (OLS). Tests for cointegration between the two assets using the ADF test. Computes the spread, rolling mean, rolling standard deviation, and z-score. Generates trading signals based on z-score thresholds. Classifies the pair's regime as "Mean Reverting" or "Trend Following" based on the average Hurst exponent. Saves plots of the spread and z-score for visualization.

*3. Main Execution:*

Data Download: Downloads historical closing prices for a predefined list of stock tickers using Yahoo Finance.
Pairs Analysis: Analyzes all possible pairs of tickers: Filters pairs based on data availability. Applies the pairs trading strategy and determines the market regime for each pair. Saves results and plots for each pair.
Summary: Saves a summary of the analyzed pairs and their regimes to a CSV file. Prints the top processed pairs and the location of saved plots.

*4. Visualization:*
   
Generates and saves plots for each pair:
Spread Plot: Displays the spread with rolling mean and standard deviation bands.
Z-Score Plot: Shows the z-score with entry/exit thresholds.

*Purpose:*

This script is a quantitative analysis tool designed to:

Identify market regimes (mean-reverting or trend-following) using statistical methods.
Implement and backtest a pairs trading strategy based on cointegration and z-score thresholds.
Visualize the results for better understanding and decision-making.

### It is useful for traders and analysts looking to explore pairs trading opportunities and classify market behavior.

