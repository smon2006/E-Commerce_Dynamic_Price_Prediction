# E-Commerce Dynamic Price Prediction

This project builds machine learning models that predict the price of a diamond from its physical and quality attributes: carat weight, cut, color, clarity, depth, table and the dimensions x, y and z. An estimate like this is the foundation of a dynamic pricing system, where an online retailer sets prices from data instead of fixed rules.

Using a dataset of 53,940 diamonds from OpenML, the project cleans the data, compares linear models with non-linear ones (Random Forest and neural networks), checks them with a custom error measure that penalises under-pricing, finds out which features drive price the most, and groups the inventory into Budget, Mid-range and Premium tiers. The best model, a Random Forest, explains about 98% of the variation in price (R² = 0.9814).

---

## Overview

Pricing too low loses revenue, and pricing too high loses customers. Dynamic pricing sets prices from data, and it relies on models that can estimate a product's fair value accurately. This project applies that idea to the diamonds dataset: it cleans the data, compares linear and non-linear models, evaluates them with a business-oriented error metric, and segments the inventory into price tiers.

## Dataset

The [diamonds dataset](https://www.openml.org/search?type=data&id=42225) from OpenML (version 1), loaded with `sklearn.datasets.fetch_openml`.

| Item | Detail |
|---|---|
| Records | 53,940 |
| Columns | 10: carat, cut, color, clarity, depth, table, price, x, y, z |
| Target | `price` (US dollars) |
| Records used after cleaning | 50,388 |

## Workflow

1. **Data loading:** the dataset is loaded from OpenML into a pandas DataFrame.
2. **Data cleaning:** 20 records with a zero value in x, y or z are removed, because a diamond cannot have a zero dimension. Dropping them is preferred to mean imputation, which would contradict the carat weight.
3. **Encoding:** cut, color and clarity are mapped to ranked integers using the standard industry hierarchies.
4. **Outlier removal:** the IQR rule on price removes 3,532 records.
5. **Exploratory analysis:** a correlation heatmap shows strong multicollinearity among carat, x, y and z.
6. **Split and scaling:** an 80/20 train-test split (random state 42), with a `StandardScaler` fitted on the training data only to prevent data leakage.
7. **Modelling:** Linear Regression (baseline), Ridge Regression, Random Forest, and two Multi-Layer Perceptron regressors.
8. **Evaluation:** MAE, RMSE, R², and a custom Asymmetric MAE that penalises under-pricing twice as heavily as over-pricing.
9. **Feature importance:** ranked importances from the Random Forest.
10. **Market segmentation:** K-Means (3 clusters) on carat and price, producing Budget, Mid-range and Premium tiers.

## Results

### Model comparison (test set, 10,078 records)

| Model | Configuration | Test R² |
|---|---|---|
| Linear Regression (baseline) | Default | 0.8987 |
| Ridge Regression | Grid search, best alpha = 100 | 0.8986 |
| Multi-Layer Perceptron | Two hidden layers of 64 neurons | 0.9767 |
| Tuned MLP | One hidden layer of 32 neurons, tanh, learning rate 0.01 | 0.9790 |
| **Random Forest** | 100 trees, maximum depth 15 | **0.9814** |

The baseline Linear Regression has an MAE of 599.44 and an RMSE of 874.36.

### Asymmetric error

Under-pricing costs a retailer more than over-pricing, so under-predictions are weighted 2x.

| Model | Asymmetric MAE |
|---|---|
| Linear Regression (baseline) | 903.63 |
| Random Forest | 307.00 |

The Random Forest reduces this error by about 66%.

### Feature importance (Random Forest)

| Rank | Feature | Importance |
|---|---|---|
| 1 | y | 0.7973 |
| 2 | carat | 0.0801 |
| 3 | clarity | 0.0722 |
| 4 | color | 0.0338 |
| 5 | x | 0.0058 |
| 6 | z | 0.0040 |
| 7 | depth | 0.0026 |
| 8 | cut | 0.0021 |
| 9 | table | 0.0020 |

The size features (y, carat, x and z) account for about 88.7% of the importance. Because they are almost perfectly correlated with each other, they should be read as a group.

### Market segmentation (Random Forest predictions)

| Segment | Diamonds | Mean carat | Mean actual price | Mean predicted price | Difference |
|---|---|---|---|---|---|
| Budget | 5,518 | 0.43 | 1,176.83 | 1,186.69 | +9.86 |
| Mid-range | 3,272 | 0.96 | 4,313.42 | 4,345.12 | +31.71 |
| Premium | 1,288 | 1.37 | 8,736.53 | 8,636.03 | −100.50 |

## Key findings

- Non-linear models are far better than linear ones (R² of 0.98 against 0.90).
- Ridge Regression gave no improvement here, because the dataset is large compared with the number of features.
- Physical size is the main driver of price. Clarity and color are secondary, and cut, depth and table add little.
- The Random Forest's average predictions are within about 1.2% of the actual average in every price tier. It slightly under-prices the Premium tier, which is the costlier kind of error.

## Limitations and future work

- The dataset has no demand, competitor-price or seasonality information, so the models estimate a diamond's value but do not react to market changes.
- The asymmetric error could be used directly as the tuning objective, so that models avoid under-pricing.
- Gradient Boosting, XGBoost and a log transformation of price could be tested.

## Repository structure

```
.
├── Project_2_E-Commerce_Dynamic_Price_Prediction.ipynb   # Notebook with code and outputs
├── E-Commerce_Dynamic_Price_Prediction_Report.pdf        # Project report
└── README.md
```

## How to run

**On Google Colab (easiest):** open Colab, go to **File → Open notebook → GitHub**, paste your repository link, pick the notebook, and choose **Runtime → Run all**.

**Locally:**

```bash
git clone https://github.com/smon2006/E-Commerce_Dynamic_Price_Prediction.git
cd E-Commerce_Dynamic_Price_Prediction
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
jupyter notebook Project_2_E-Commerce_Dynamic_Price_Prediction.ipynb
```

An internet connection is needed to download the dataset from OpenML. The notebook runs in a few minutes in total, and the MLP grid search in Question 14 is the slowest step.

## Tools

Python, pandas, NumPy, Matplotlib, Seaborn and scikit-learn (Linear Regression, Ridge, Random Forest, MLPRegressor, KMeans, GridSearchCV), run in Google Colab.

## References

- OpenML. *Diamonds dataset* (version 1). https://www.openml.org/search?type=data&id=42225
- Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.
- Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5–32.
- Hoerl, A. E. and Kennard, R. W. (1970). Ridge Regression: Biased Estimation for Nonorthogonal Problems. *Technometrics*, 12(1), 55–67.
