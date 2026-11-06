# Data

**Source:** [Student Performance Factors](https://www.kaggle.com/datasets/lainguyn123/student-performance-factors) on Kaggle

**File:** `raw/StudentPerformanceFactors.csv`, 6,607 rows x 20 columns (6,606 after removing one score above 100)

## Columns

### Numerical

| Column | Description | Range |
|---|---|---|
| `Hours_Studied` | Hours studied per week | 1-44 |
| `Attendance` | Percentage of classes attended | 60-100 |
| `Sleep_Hours` | Average sleep per night | 4-10 |
| `Previous_Scores` | Score in previous exams | 50-100 |
| `Tutoring_Sessions` | Tutoring sessions per month | 0-8 |
| `Physical_Activity` | Hours of physical activity per week | 0-6 |
| `Exam_Score` | **Target.** Final exam score | 55-100 |

### Categorical

| Column | Levels |
|---|---|
| `Parental_Involvement` | Low, Medium, High |
| `Access_to_Resources` | Low, Medium, High |
| `Extracurricular_Activities` | No, Yes |
| `Motivation_Level` | Low, Medium, High |
| `Internet_Access` | No, Yes |
| `Family_Income` | Low, Medium, High |
| `Teacher_Quality` | Low, Medium, High |
| `School_Type` | Public, Private |
| `Peer_Influence` | Negative, Neutral, Positive |
| `Learning_Disabilities` | No, Yes |
| `Parental_Education_Level` | High School, College, Postgraduate |
| `Distance_from_Home` | Near, Moderate, Far |
| `Gender` | Male, Female |

## Data Quality

- **Invalid value:** one row has `Exam_Score = 101` and is removed.
- **Missing values:** `Teacher_Quality` (78), `Parental_Education_Level` (90) and `Distance_from_Home` (67). They are filled with the most frequent level inside the model pipeline.
- **Duplicates:** none.
