# Machine Learning Project 1

Two supervised-learning projects built as Kaggle-style competition entries for the Machine Learning course at IIIT Bangalore (M.Tech CSE). Each project includes the working notebooks, the submission files that were uploaded to Kaggle, and a full written report.

| Project | Task | Best public Kaggle score |
|---|---|---|
| [Domestic Flight Price Prediction](domestic-flight-price-prediction/) | Regression: predict ticket price (Rs) | **0.96238** |
| [Vehicle Type Classification](vehicle-type-classification/) | 3-class classification from 20 numeric specs | **0.90500** |

## Team

- Param Pandey (MT2026065)
- Nabodaya Mete (MT2026058)
- Prakhar Srivastava (MT2026071)
- Varun Gupta (MT2026031)

## Repository Structure

```
ml-project-1/
├── domestic-flight-price-prediction/
│   ├── notebooks/       # preprocessing, modelling, tuning, and consolidated final notebook
│   ├── reports/         # Flight_Price_Prediction_LaTeX.pdf
│   ├── submissions/     # Kaggle submission CSVs
│   └── dataset/
└── vehicle-type-classification/
    ├── notebooks/       # Vehicle_Classification_Final_Course_Project.ipynb
    ├── reports/         # ML_Proj_Vehicle_Classification_Report.pdf
    ├── submissions/     # final five ranked Kaggle submission CSVs
    └── dataset/         # train / test / sample-submission CSVs
```

---

## 1. Domestic Flight Price Prediction

Predict the fare of a domestic flight from its airline code, route, departure/arrival time, stops, cabin class, duration and days left before departure.

**Data:** 15,589 training flights and 3,898 test flights (city and airline names are anonymised).

### Approach

- **Target transform:** models predict `log1p(price)` and are mapped back with `expm1`. This tames an ~85x price range and turns errors into relative errors.
- **Missing values:**
  - `airline` (about 20% missing) was recovered from the 2-letter flight-code prefix, which identifies the airline with 100% purity.
  - `duration` (about 20% missing) was filled in two ways. Round 1 used a hierarchical median over flight + route + stops. Round 2 used an Extra Trees model (OOF MAE 0.109 h).
- **Encoding:** ordinal for `stops`, binary for `flight_class`, one-hot for time bands, flight prefix and *directional* route (A→B differs from B→A). This gives 65 features in round 1.
- **De-duplication:** dropping the flight number created exact duplicate rows, which were removed from training.
- **Models compared (two preprocessing rounds):** Ridge, Random Forest, Extra Trees, Hist Gradient Boosting, LightGBM, XGBoost, CatBoost, plus voting ensembles.
- **Validation:** 80/20 hold-out stratified by class, plus 5-fold cross-validation.

### Final Model

An equal-weight `VotingRegressor(XGBoost, ExtraTrees)` trained on round-1 data.

| Metric (5-fold CV) | Value |
|---|---|
| log RMSE | 0.1352 |
| R² (log price) | 0.9817 |
| MAE | Rs 3,339 |
| MAPE | 7.9% |
| **Public Kaggle score** | **0.96238** |

### Key Findings

- Cabin class explains most of the variance (a class-mean baseline alone reaches R² 0.764 on log-price).
- Domain-driven cleaning (airline from prefix, directional routes) beat clever imputation: the ML-based duration fill looked better offline but scored lower on Kaggle (0.96077).
- A diverse blend beat both of its members and beat hyper-parameter tuning of single models.
- Tuning for raw-price R² made XGBoost worse on log RMSE, MAE and MAPE, so the optimisation metric matters.
- PCA and k-means found only two natural groups (short-haul Economy vs long-haul/connecting Business); adding the cluster label did not improve the blend.

### Notebooks

| Notebook | Purpose |
|---|---|
| `flight_preprocessing.ipynb`, `flight_preprocessing2.ipynb` | Round 1 and round 2 preprocessing |
| `flight_ML.ipynb`, `flight_ML2.ipynb` | Model comparison on each round |
| `flight_ml3.ipynb` | Randomized-search tuning of the two blend members |
| `Flight_Price_Prediction_Final.ipynb` | Consolidated notebook that regenerates every figure and the final predictions |
| `flight_price_prediction_aaukat.ipynb`, `flight_price_prediction_complete.ipynb` | Additional experiments |

---

## 2. Vehicle Type Classification

Classify a vehicle into one of three encoded classes (0, 1, 2) from 20 numeric specifications covering powertrain, geometry, clearance, energy/aerodynamics and driving dynamics.

**Data:** 10,000 training rows (7,500 unique feature vectors after removing 2,500 consistent duplicates) and 2,500 test rows. There are no missing values and no categorical features.

### Approach

- **Leakage control:** duplicates were removed before folding, and scalers were fitted inside each CV fold.
- **Validation:** shuffled 5-fold `StratifiedKFold`, repeated over seeds 42, 2026 and 17, with out-of-fold accuracy, macro-F1, class-0 recall and log loss.
- **Models screened:** LDA, QDA, Gaussian NB, Ridge/Lasso least-squares, L1/L2/Elastic Net multinomial Logistic Regression, Decision Tree, Random Forest and Extra Trees.
- **Feature work:** dropping `city_fuel_efficiency_kmpl` helped the simple model. Eight engineered ratio features (e.g. horsepower/weight, torque/weight, body proportions) were tested. Polynomial and interaction features showed no stable gain.
- **Threshold tuning:** per-class bias offsets did not generalise, so the final models use plain argmax.

### Final Submissions

Penalized multinomial logistic models outperformed most Gaussian and tree-based alternatives. The five final Kaggle entries, in priority order:

| # | File | Model | Public score |
|---|---|---|---|
| 1 | `top01_ensemble_905.csv` | 50/50 drop-city L2 Logistic + full-feature Elastic Net | **0.90500** |
| 2 | `top02_dropcity_904.csv` | 19-feature L2 multinomial Logistic Regression | 0.90400 |
| 3 | `top03_elastic_904.csv` | 20-feature Elastic Net Logistic Regression | 0.90400 |
| 4 | `top04_engineered_903.csv` | Logistic Regression + 8 engineered features | 0.90300 |
| 5 | `top05_lda20_902.csv` | 40/40/20 Logistic / Elastic Net / LDA blend | 0.90200 |

### Key Findings

- Of 792 out-of-fold errors, 772 involve class 0; confusion between classes 1 and 2 is minimal.
- Repeated-CV differences between the finalists are tiny (about 89.3% each), so the shortlist favours model diversity over leaderboard chasing.
- Training on 20% of the data (1,200 rows) gave 88.07% validation accuracy vs 88.87% on 100%.
- PCA was used for visualization and exploration only; it did not improve accuracy.

---

## Getting Started

### Prerequisites

- Python 3.9+
- Jupyter Notebook or Google Colab

Install the core libraries:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn xgboost lightgbm catboost jupyter
```

### Running the Notebooks

```bash
git clone https://github.com/halconie/ml-project-1.git
cd ml-project-1
jupyter notebook
```

Then open the consolidated notebook for the project you want to reproduce:

- `domestic-flight-price-prediction/notebooks/Flight_Price_Prediction_Final.ipynb`
- `vehicle-type-classification/notebooks/Vehicle_Classification_Final_Course_Project.ipynb`

**Note:** the flight project's original `train.csv` and `test.csv` come from the course Kaggle competition and are not included in this repository, so place them where the notebook expects them before running. The vehicle dataset is included under `vehicle-type-classification/dataset/`.

## Reports

Full write-ups, including preprocessing, EDA, model comparison, error analysis, limitations and future work, are in each project's `reports/` folder:

- [Flight Price Prediction report](domestic-flight-price-prediction/reports/Flight_Price_Prediction_LaTeX.pdf)
- [Vehicle Classification report](vehicle-type-classification/reports/ML_Proj_Vehicle_Classification_Report.pdf)

## Notes

- All reported Kaggle scores are **public leaderboard** scores. Private-test scores were not available when the reports were written.
- The `dataset/` folder inside `domestic-flight-price-prediction/` contains a vehicle-classification submission bundle rather than flight data, so you may want to move or remove it.
- No license file is currently included.
