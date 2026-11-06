import joblib
import gradio as gr
import pandas as pd

from src.features import add_engineered_features

MODEL_PATH = 'models/best_model.pkl'
AT_RISK_SCORE = 63  # Bottom 10 % of exam scores in the dataset
AVERAGE_SCORE = 67.2

model = joblib.load(MODEL_PATH)


FEATURES = ['Hours_Studied', 'Attendance', 'Parental_Involvement', 'Access_to_Resources',
            'Extracurricular_Activities', 'Sleep_Hours', 'Previous_Scores', 'Motivation_Level',
            'Internet_Access', 'Tutoring_Sessions', 'Family_Income', 'Teacher_Quality', 'School_Type',
            'Peer_Influence', 'Physical_Activity', 'Learning_Disabilities', 'Parental_Education_Level',
            'Distance_from_Home', 'Gender']

LOW_MEDIUM_HIGH = ['Low', 'Medium', 'High']
NO_YES = ['No', 'Yes']


def predict_score(*values):
    student = pd.DataFrame([dict(zip(FEATURES, values))])
    score = model.predict(student)[0]
    score = min(max(score, 0), 100)

    if score < AT_RISK_SCORE:
        status = "**At risk.** This student is predicted to be in the bottom 10% and may need extra support."
    elif score < AVERAGE_SCORE - 1:
        status = "Below the average score of 67."
    elif score <= AVERAGE_SCORE + 1:
        status = "Around the average score of 67."
    else:
        status = "Above the average score of 67."

    return f"## Predicted exam score: {score:.1f} / 100\n\n{status}"


CSS = """
.gradio-container {max-width: 900px !important; width: 100% !important; margin: auto !important;}
#title {text-align: center;}
"""

with gr.Blocks(title='Student Exam Score Predictor') as demo:
    gr.Markdown(
        "# Student Exam Score Predictor\n"
        "Fill in the student's details and click **Predict** to see the expected exam score.",
        elem_id='title',
    )

    inputs = {}

    gr.Markdown("### Study Habits")
    inputs['Hours_Studied'] = gr.Slider(1, 44, value=20, step=1, label='Hours studied per week')
    inputs['Attendance'] = gr.Slider(60, 100, value=80, step=1, label='Attendance (%)')
    inputs['Previous_Scores'] = gr.Slider(50, 100, value=75, step=1, label='Previous exam score')
    inputs['Tutoring_Sessions'] = gr.Slider(0, 8, value=1, step=1, label='Tutoring sessions per month')
    inputs['Extracurricular_Activities'] = gr.Radio(NO_YES, value='Yes', label='Extracurricular activities')

    gr.Markdown("### Health and Lifestyle")
    inputs['Sleep_Hours'] = gr.Slider(4, 10, value=7, step=1, label='Sleep hours per night')
    inputs['Physical_Activity'] = gr.Slider(0, 6, value=3, step=1, label='Physical activity (hours per week)')

    gr.Markdown("### Personal")
    inputs['Motivation_Level'] = gr.Radio(LOW_MEDIUM_HIGH, value='Medium', label='Motivation level')
    inputs['Learning_Disabilities'] = gr.Radio(NO_YES, value='No', label='Learning disabilities')
    inputs['Gender'] = gr.Radio(['Male', 'Female'], value='Male', label='Gender')

    gr.Markdown("### Family")
    inputs['Parental_Involvement'] = gr.Radio(LOW_MEDIUM_HIGH, value='Medium', label='Parental involvement')
    inputs['Parental_Education_Level'] = gr.Radio(['High School', 'College', 'Postgraduate'],
                                                  value='High School', label='Parental education')
    inputs['Family_Income'] = gr.Radio(LOW_MEDIUM_HIGH, value='Low', label='Family income')

    gr.Markdown("### School and Environment")
    inputs['Teacher_Quality'] = gr.Radio(LOW_MEDIUM_HIGH, value='Medium', label='Teacher quality')
    inputs['School_Type'] = gr.Radio(['Public', 'Private'], value='Public', label='School type')
    inputs['Peer_Influence'] = gr.Radio(['Negative', 'Neutral', 'Positive'], value='Positive', label='Peer influence')
    inputs['Access_to_Resources'] = gr.Radio(LOW_MEDIUM_HIGH, value='Medium', label='Access to resources')
    inputs['Internet_Access'] = gr.Radio(NO_YES, value='Yes', label='Internet access')
    inputs['Distance_from_Home'] = gr.Radio(['Near', 'Moderate', 'Far'], value='Near', label='Distance from home')

    predict_button = gr.Button('Predict', variant='primary', size='lg')
    output = gr.Markdown()

    predict_button.click(predict_score, inputs=[inputs[f] for f in FEATURES], outputs=output)


if __name__ == '__main__':
    demo.launch(css=CSS)
