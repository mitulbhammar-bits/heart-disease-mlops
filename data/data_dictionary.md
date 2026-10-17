# Data dictionary - original Cleveland encoding

13 predictors plus binary target; the downloaded original target has codes 0..4.
All friendly columns preserve original values, not Kaggle-remapped values.

| Friendly name | UCI name | Meaning and coding |
|---|---|---|
| age | age | Age in years |
| sex | sex | Recorded sex: 0 female, 1 male |
| chest_pain_type | cp | 1 typical angina, 2 atypical angina, 3 non-anginal pain, 4 asymptomatic |
| resting_blood_pressure | trestbps | Resting blood pressure, mm Hg |
| cholesterol | chol | Serum cholesterol, mg/dl |
| high_fasting_blood_sugar | fbs | Fasting blood sugar >120 mg/dl: 1 true, 0 false |
| resting_ecg | restecg | 0 normal, 1 ST-T abnormality, 2 probable/definite LV hypertrophy |
| maximum_heart_rate | thalach | Maximum achieved heart rate |
| exercise_angina | exang | Exercise-induced angina: 1 yes, 0 no |
| st_depression | oldpeak | Exercise-induced ST depression relative to rest |
| st_slope | slope | 1 upsloping, 2 flat, 3 downsloping |
| major_vessels | ca | Number of major vessels colored by fluoroscopy: 0..3; missing allowed |
| thalassemia_test | thal | UCI thal test: 3 normal, 6 fixed defect, 7 reversible defect; missing allowed |
| heart_disease | num | 0 absent; original codes 1..4 combined to 1 present |

`?` becomes a blank in the cleaned CSV. This is intentional: the model pipeline
imputes within training folds to avoid leakage. Do not globally fill the full CSV
before CV. The correlation heatmap uses codes for descriptive purposes; categorical
code correlations are not evidence of linear or causal relationships.

Source: https://archive.ics.uci.edu/dataset/45/heart+disease
License: CC BY 4.0. Citation: https://doi.org/10.24432/C52P4X
