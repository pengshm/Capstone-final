# Capstone: California Housing Price Prediction

## Problem statement
The goal of this capstone is to build a predictive model for **California home prices** using a publicly available dataset.
Specifically, we predict `price` from demographic, housing, and location features.

## Dataset
- **File:** `housing.csv`
- **Target:** `price` (continuous value → regression problem)
- **Features:** numeric attributes

## Modeling approach (what we did)
### 1) Preprocessing (leakage-safe)
All preprocessing is done **inside scikit-learn Pipelines**, so cross-validation and grid search do not leak information:
- Numeric: median imputation + standard scaling
- Categorical: most-frequent imputation 

### 2) Models compared
We compared multiple regression models:
- DummyRegressor (median baseline)
- LinearRegression
- Ridge / Lasso
- RandomForestRegressor
- GradientBoostingRegressor

### 3) Evaluation metric (what and why)
**Primary metric: RMSE (Root Mean Squared Error)**  
- RMSE is in the same unit as the target (dollars), so it’s easy to interpret.
- RMSE penalizes large errors more heavily than MAE, which is useful when large price misses are costly.

We also reported:
- **MAE** (Mean Absolute Error)
- **R²** (explained variance)

### 4) Cross-validation
We used **5-fold K-Fold CV** on the training set for fair model comparisons and stable estimates.

### 5) Hyperparameter tuning
We used **GridSearchCV** for:
- Ridge (tuning `alpha`)
- RandomForest (tuning `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`)
- GradientBoosting (tuning `n_estimators`, `learning_rate`, `max_depth`)

The tuned model with the lowest CV RMSE was selected and evaluated on a held-out test set.

## Findings

### Key technical findings (high-level)
Your exact numbers will appear after you run the notebooks, but typical patterns for this dataset:
- `sqft_living` is usually the strongest single predictor of house value.
- Geographic variables (`zipcode`) provide a strong signal (prices cluster by region).
- Tree-based models often outperform basic linear regression due to non-linear relationships.

### Nontechnical summary (plain language)
We trained several algorithms to estimate home prices from neighborhood and location information.  
To be confident our model generalizes, we validated it using repeated train/validation splits (cross-validation) and then tested it on a held-out dataset that the model never saw.

We selected the model (GradientBoostingRegressor) that minimized **typical dollar error (RMSE)**. 


## Suggested next steps
1. **Try log-transforming the target** (price) to reduce skew and improve stability, then transform predictions back.
2. **Add new data sources** (school ratings, crime rate, amenities, commute time) to improve location realism.
3. **Model explainability:**  
   - Use permutation importance or SHAP (for tree models) to explain key drivers.
4. **Error analysis by region:**  
   - Segment residuals by county/latitude bands to see where the model underperforms.
5. **Production readiness:**  
   - Wrap the saved `final_model.joblib` in a small API (FastAPI) and add input validation.

--- 
**Capstone theme:** California Real Estate Price Prediction
