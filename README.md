# Student Performance Analysis

Predicting students' exam scores from study habits, family background and school environment, and finding out which factors actually matter.

**Result:** a scikit-learn Linear Regression pipeline that predicts 98.7% of test students within ±1 point of their real score (test R² = 0.83, MAE = 0.42). A Gradio web app lets anyone enter a student's details and get a prediction.

## Research Questions

1. Which factors are most strongly linked to exam scores?
2. Do controllable factors (study hours, attendance) matter more than background factors (income, parental education)?
3. How accurately can a model predict a student's score?
4. Are there unexpected patterns or anomalies in the data?

## Dataset

[Student Performance Factors](https://www.kaggle.com/datasets/lainguyn123/student-performance-factors) (Kaggle): 6,607 students, 19 predictors (6 numerical, 13 categorical) and the target `Exam_Score`. See [data/README.md](data/README.md) for the column list.

## Approach

1. **Cleaning:** removed one impossible score (101). Missing values (about 1% in three columns) are imputed inside the model pipeline.
2. **EDA:** distributions, IQR outliers, correlations, one-way ANOVA with effect sizes (η²).
3. **Feature engineering:** four ratio and combined features (`src/features.py`).
4. **Modeling:** one scikit-learn `Pipeline` (feature engineering, imputation, ordinal / one-hot encoding, scaling, model), so there is no data leakage and the saved model accepts raw data.
5. **Evaluation:** 5-fold cross-validation, `GridSearchCV` tuning, best model chosen by CV R² and reported once on a held-out 20% test set.

## Results

| Model | CV R² (mean ± std) | Test R² | Test RMSE | Test MAE |
|---|---|---|---|---|
| **Linear Regression** | **0.715 ± 0.074** | **0.826** | **1.52** | **0.42** |
| Neural Network (tuned) | 0.654 ± 0.079 | 0.763 | 1.77 | 0.85 |
| Neural Network | 0.645 ± 0.064 | 0.740 | 1.86 | 0.94 |
| Decision Tree (tuned) | 0.530 ± 0.045 | 0.622 | 2.23 | 1.41 |
| Decision Tree | 0.111 ± 0.048 | 0.163 | 3.33 | 1.63 |

## Key Findings

- **Attendance and study hours dominate.** Attendance (r = 0.58) and Hours_Studied (r = 0.45) are the top predictors. Previous scores are surprisingly weak (r = 0.17).
- **Controllable factors matter most.** The top and bottom quarter of students differ by 5.8 points on attendance and 4.2 on study hours. Every background factor shifts the score by under 2 points.
- **No effect** from gender, school type, sleep or physical activity.
- **Anomaly:** 49 students (0.7%) score 15-30 points above their prediction, while nobody scores more than about 6 points below it. Nothing in the recorded features explains them.

## Project Structure

```
├── app.py                        # Gradio web app
├── data/
│   ├── raw/StudentPerformanceFactors.csv
│   └── README.md                 # column descriptions
├── models/
│   └── best_model.pkl            # trained pipeline (created by the notebook)
├── notebooks/
│   └── Student_performance_factors.ipynb
├── src/
│   └── features.py               # feature engineering used by the pipeline
├── requirements.txt
└── README.md
```

## Getting Started

Requires Python 3.11 or newer.

```bash
git clone https://github.com/Tanvir-h-simon/Student-Performance-Analysis.git
cd Student-Performance-Analysis

python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
```

### Run the web app

```bash
python app.py
```

Then open http://127.0.0.1:7860. While editing the UI, `gradio app.py` reloads the app automatically on every save.

### Retrain the model

Open `notebooks/Student_performance_factors.ipynb` and run all cells. The last cell saves the trained pipeline to `models/best_model.pkl`, which the app loads.

## Using the Saved Model in Python

Run from the project root. `src.features` must be importable because the pipeline calls `add_engineered_features`.

```python
import joblib
import pandas as pd
from src.features import add_engineered_features  # needed to load the pipeline

model = joblib.load('models/best_model.pkl')

students = pd.read_csv('data/raw/StudentPerformanceFactors.csv').drop(columns='Exam_Score')
print(model.predict(students.head()))
```

The model only loads with the scikit-learn version that saved it (1.9.1, pinned in `requirements.txt`).

## Tech Stack

Python, pandas, NumPy, Matplotlib, seaborn, SciPy, scikit-learn, Gradio

## Author

Tanvir Hossain ([GitHub](https://github.com/Tanvir-h-simon))
