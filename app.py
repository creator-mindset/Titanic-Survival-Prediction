
import streamlit as st
import joblib
import numpy as np

model = joblib.load('titanic_model.pkl')
st.title('Titanic Survival Prediction')
st.write('Enter the details of the passenger')
pclass = st.selectbox(
    'Passenger Class',
    [1, 2, 3]
)

sex = st.selectbox(
    'Gender',
    ['Male', 'Female']
)

age = st.slider(
    'Age',
    min_value=0,
    max_value=100
)

sibsp = st.slider(
    'Number of Siblings/Spouse',
    min_value=0,
    max_value=10
)



if pclass == 1:
    fare = st.slider(
        'Fare (in $)',
        min_value=50,
        max_value=600,
        value=100
    )

elif pclass == 2:
    fare = st.slider(
        'Fare (in $)',
        min_value=20,
        max_value=150,
        value=50
    )

else:
    fare = st.slider(
        'Fare (in $)',
        min_value=5,
        max_value=80,
        value=20
    )

sex_value = 0 if sex == 'Female' else 1

if st.button('Predict'):

    data_input = np.array([
        [pclass, sex_value, age, sibsp, fare]
    ])

    prediction = model.predict(data_input)

    probability = model.predict_proba(data_input)

    survival_chance = probability[0][1] * 100

  
    if prediction[0] == 1:

        st.success(
            f'Passenger Survived ✅\n\n'
            f'Survival Chance: {survival_chance:.2f}%'
        )

        st.balloons()

    else:

        st.error(
            f'Passenger Did Not Survive ❌\n\n'
            f'Survival Chance: {survival_chance:.2f}%'
        )

        st.snow()
