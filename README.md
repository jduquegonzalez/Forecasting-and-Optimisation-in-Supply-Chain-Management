# Forecasting and Optimisation in Supply Chain Management

This repository contains a collection of Jupyter notebooks demonstrating various data science techniques for forecasting and optimisation in supply chain management. The notebooks leverage concepts and code examples primarily from Nicolas Vandeput's book "Data Science for Supply Chain Forecasting."

## Notebooks

This repository includes the following Jupyter notebooks:

### 1. Regression Tree Model - Single and Optimised (1 feature)
This notebook explores the use of regression trees for forecasting. It covers:
* Importing and preprocessing sales data.
* Creating training and test datasets.
* Building an initial regression tree model.
* Evaluating the model using KPIs like MAE, RMSE, and Bias.
* Optimising the regression tree parameters using K-fold Cross-validation and Random Search to improve accuracy and avoid overfitting.
* Evaluating the optimised model.

### 2. Neural Network Regression Model for Sales Forecasting
This notebook demonstrates how to build and optimise a neural network regression model for car sales forecasting. Key steps include:
* Importing necessary libraries like MLPRegressor, RandomizedSearchCV, pandas, numpy, and StandardScaler.
* Data import and preprocessing, including formatting data into a period-wise structure.
* Preparing training and testing datasets.
* Scaling the data using StandardScaler.
* Defining a function to calculate Key Performance Indicators (MAE, RMSE, Bias).
* Configuring and training an initial MLPRegressor model.
* Evaluating the initial model.
* Defining an enhanced hyperparameter grid and using RandomizedSearchCV for tuning the MLPRegressor.
* Evaluating the optimised neural network model.
* Generating forecasts using the final optimised model.

### 3. Linear Regression Analysis of Automobile Sales Data
This notebook focuses on applying linear regression for analysing and forecasting automobile sales data. It includes:
* Importing and transforming automobile sales data into a pivot table format.
* Creating training and test datasets.
* Building and training a linear regression model.
* Generating predictions and evaluating the model's accuracy using KPIs.
* Making future forecasts with the trained model.

### 4. K-Means Clustering Explained
This notebook explains and implements K-means clustering, an unsupervised learning technique, for classifying unlabelled data into groups. The process involves:
* Importing and preparing sales data.
* Discussing feature selection for clustering (e.g., volume, additive/multiplicative seasonality).
* Scaling data, as K-means is sensitive to extreme values.
* Computing multiplicative seasonal factors.
* Defining and applying a scaler function to normalise seasonal factors.
* Using the KMeans algorithm from scikit-learn to identify clusters.
* Experimenting with the number of clusters by analysing inertia (sum of squared distances to cluster centroids) using the elbow method.
* Visualising cluster centres using a heatmap.
* Counting the number of products within each cluster.

### 5. Extreme Gradient Boosting Optimisation (single and multiple output)
This notebook explores Extreme Gradient Boosting (XGBoost), a powerful machine learning algorithm. Key aspects covered are:
* Importing and preparing sales data.
* Creating training and test datasets.
* Initialising and fitting an XGBoost regression model.
* Analysing feature importance.
* Understanding and implementing evaluation sets and early stopping to prevent overfitting and reduce training time.
* Using a holdout dataset as an evaluation dataset.
* Hyperparameter tuning using RandomizedSearchCV with a defined parameter grid.
* Training the XGBoostRegressor with optimised parameters.
* Evaluating the optimised model.
* Implementing XGBoost for multiple period forecasting using `MultiOutputRegressor`.

### 6. Forest Model Analysis (Ensemble Models)
This notebook delves into ensemble models, specifically Random Forests, leveraging the "wisdom of the crowd" concept for improved prediction accuracy. It includes:
* Importing and preparing sales data.
* Creating training and test datasets.
* Training an initial Random Forest Regressor model, explaining techniques to create diverse trees (limiting features per split, bootstrapping samples).
* Evaluating the unoptimised Random Forest model.
* Optimising Random Forest parameters using K-fold Cross-validation and Random Search.
* Evaluating the optimised model and displaying its accuracy.
* Plotting feature importance from the optimised model.

### 7. Leveraging Categorical Features in Supply Chain Forecasting
This notebook demonstrates how to incorporate categorical features (e.g., market, product family) into machine learning models to potentially improve forecast accuracy. The key steps are:
* Importing and preparing sales data.
* Updating the dataset creation function to handle categorical columns, identified by a prefix separator in their names.
* Using one-hot label encoding (dummification) to represent categorical features like 'Brand'.
* Preparing training and test sets that include these encoded categorical features.

### 8. Enhancing Time Series Forecasts with Leading Indicators in Python
This notebook focuses on incorporating external and internal factors (leading indicators) like GDP growth, price changes, or promotions into time series forecasting models. It covers:
* Importing and transforming car sales data.
* Retrieving and structuring economic data (e.g., GDP) from an API for use as an exogenous variable.
* Defining a function (`datasets_exo`) to prepare datasets that include these exogenous variables alongside historical demand and month information.
* This prepared data can then be used in various machine learning models.

## How to Use
1.  Clone the repository to your local machine.
2.  Ensure you have Python and the necessary libraries installed (pandas, numpy, scikit-learn, xgboost, openpyxl, requests, seaborn). You can typically install these using pip:
    ```bash
    pip install pandas numpy scikit-learn xgboost openpyxl requests seaborn matplotlib
    ```
3.  Open the Jupyter notebooks using Jupyter Notebook or JupyterLab to explore the code and explanations.
4.  The data used in these notebooks is primarily sourced from `https://supchains.com/wp-content/uploads/2021/07/norway_new_car_sales_by_make1.csv` and economic data from `https://data.ssb.no/api/v0/en/table/09189/`.

This repository serves as a practical guide and learning resource for applying data science techniques to common challenges in supply chain management.
