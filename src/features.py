LEVEL_TO_NUMBER = {'Low': 1, 'Medium': 2, 'High': 3}
MAX_TUTORING_SESSIONS = 8

ENGINEERED_FEATURES = ['Study_Efficiency', 'Support_Score', 'Total_Resources', 'Study_Life_Balance']


def add_engineered_features(df):
    """Return a copy of df with the four engineered columns added."""
    df = df.copy()

    # Previous score per hour studied
    df['Study_Efficiency'] = df['Previous_Scores'] / (df['Hours_Studied'] + 1) # +1 avoids division by zero

    # Parental involvement (1-3) + tutoring sessions rescaled to 0-3.
    parental = df['Parental_Involvement'].map(LEVEL_TO_NUMBER).fillna(2) # Missing involvement counts as Medium.
    df['Support_Score'] = parental + df['Tutoring_Sessions'] / MAX_TUTORING_SESSIONS * 3

    # Access to resources (1-3) + internet access (0/1)
    resources = df['Access_to_Resources'].map(LEVEL_TO_NUMBER).fillna(2)
    internet = df['Internet_Access'].map({'No': 0, 'Yes': 1}).fillna(0)
    df['Total_Resources'] = resources + internet

    # Sleep hours per study hour
    df['Study_Life_Balance'] = df['Sleep_Hours'] / (df['Hours_Studied'] + 1)

    return df