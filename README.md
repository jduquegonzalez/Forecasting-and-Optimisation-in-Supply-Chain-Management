# Forecasting the Popularity of "Flowers" Searches on Google: A Comparative Analysis of Prophet, SARIMA, and Holt-Winters Models
This project is adapted from Marco Peixeiro's book [Time Series Forecasting in Python](https://amzn.eu/d/eU0GuYo).

## Introduction

Accurately predicting the popularity of search terms is crucial for retail analysts, as it enables more effective optimisation of marketing strategies. By understanding the trends in keyword searches, analysts can help marketing teams refine their bidding strategies, thus influencing the cost-per-click on advertisements and ultimately improving the overall return on investment (ROI) for marketing campaigns.

Moreover, forecasting search term trends offers deeper insights into consumer behaviour. For instance, if we can predict an increase in searches for "flowers" in the coming month, retail businesses can proactively adjust their strategies. This could involve offering timely discounts, adjusting inventory levels, and ensuring sufficient stock to meet the anticipated demand. Such insights help in aligning supply with demand, improving customer satisfaction, and driving sales growth.

In this project, we employ the SARIMA, Holt-Winters, and Prophet models to predict future trends in "flowers" searches on Google. By analysing historical search data sourced from [Google Trends](https://trends.google.com/trends/explore?date=all&geo=GB&q=%2Fm%2F0c9ph5), we aim to create a precise model that provides actionable insights for retail analysts, enabling data-driven decision-making to enhance marketing effectiveness and operational efficiency.

## Project Structure

- `Forecasting the Popularity of "Flowers" Searches on Google.ipynb`: The main Jupyter notebook containing the analysis and model implementation.
- `data/`: Directory to store the raw and processed data (not included in the repository for privacy reasons).
- `figures/`: Directory to store the generated figures and plots.
- `forecast_results/`: Directory to store the results.

## Dependencies

- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- statsmodels
- fbprophet (Prophet)
- tqdm
- scikit-learn


You can install the dependencies using pip:

```sh
pip install pandas numpy matplotlib seaborn statsmodels fbprophet tqdm scikit-learn
```

## Usage

1. Clone the repository:

```sh
git clone https://github.com/jduquegonzalez/Supply-Chain-Efficiency-and-Forecasting.git
```

2. Navigate to the project directory:

```sh
cd Supply-Chain-Efficiency-and-Forecasting
```

3. Open the Jupyter notebook:

```sh
jupyter notebook "Forecasting the Popularity of "Flowers" Searches on Google.ipynb"
```

4. Run the notebook cells to reproduce the analysis and results.

## Methodology

### Data Preprocessing

- Importing and cleaning the data.
- Handling missing values and outliers.

### Model Implementation

- **SARIMA**: Seasonal ARIMA model for capturing the seasonality and trend in the time series.
- **Holt-Winters**: Exponential Smoothing model to capture level, trend, and seasonal components.
- **Prophet**: General additive model developed by Facebook, combining trend, seasonality, and holiday effects.

### Model Evaluation

- Evaluating model performance using Mean Absolute Error (MAE) and Mean Squared Error (MSE).
- Comparing the performance of SARIMA, Holt-Winters, and Prophet models.

### Forecasting

- Generating future forecasts using the best performing model.
- Plotting the forecast with confidence intervals.

## Conclusion

This project demonstrates the application of time series forecasting techniques to predict the popularity of search terms on Google. By leveraging models like SARIMA, Holt-Winters, and Prophet, we can gain valuable insights into search trends, aiding retail analysts in making informed decisions to optimise marketing strategies and inventory management.

## References

- Marco Peixeiro, [Time Series Forecasting in Python](https://amzn.eu/d/eU0GuYo)
- [Google Trends](https://trends.google.com/trends/explore?date=all&geo=GB&q=%2Fm%2F0c9ph5)
- Nicolas Vandeput, [Inventory Optimization: Models and Simulations](https://amzn.eu/d/0VEXlHI)
- Jordan Ellenberg, [How Not to Be Wrong: The Power of Mathematical Thinking](https://amzn.eu/d/edXzDHa)

---

Feel free to contribute to this project by opening issues or submitting pull requests.
```
