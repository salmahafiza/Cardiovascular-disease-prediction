import re
import streamlit as st
import pandas as pd
import numpy as np
import cv2
import pytesseract
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')
import firebase_admin
from firebase_admin import credentials, auth


# Set the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  
# Update this with your Tesseract executable path


# Initialize Firebase
keypath = "C:\\Users\\zee7s\\OneDrive\\Desktop\\project\\health-57aaf-acfa7dee0168.json"
cred = credentials.Certificate(keypath)
if not firebase_admin._apps:
    firebase_admin.initialize_app(cred)

# Function to preprocess the data
def preprocess_data(df, is_user_input=False):
    df.rename(columns={'cigsPerDay':'cigs_per_day','BPMeds':'bp_meds',
                       'prevalentStroke':'prevalent_stroke','prevalentHyp':'prevalent_hyp',
                       'totChol':'total_cholesterol','sysBP':'systolic_bp','diaBP':'diastolic_bp',
                       'BMI':'bmi','heartRate':'heart_rate','TenYearCHD':'ten_year_chd'},
              inplace=True)
    
    if not is_user_input:
        if 'sex' in df.columns:
            df['sex'] = np.where(df['sex'] == 'M', 1, 0)
        if 'is_smoking' in df.columns:
            df['is_smoking'] = np.where(df['is_smoking'] == 'YES', 1, 0)
        
        if 'education' in df.columns:
            df['education'] = df['education'].fillna(df['education'].mode()[0])
        if 'bp_meds' in df.columns:
            df['bp_meds'] = df['bp_meds'].fillna(df['bp_meds'].mode()[0])
        if 'cigs_per_day' in df.columns and 'is_smoking' in df.columns:
            df['cigs_per_day'] = df['cigs_per_day'].fillna(df[df['is_smoking'] == 1]['cigs_per_day'].median())
        if 'total_cholesterol' in df.columns:
            df['total_cholesterol'] = df['total_cholesterol'].fillna(df['total_cholesterol'].median())
        if 'bmi' in df.columns:
            df['bmi'] = df['bmi'].fillna(df['bmi'].median())
        if 'heart_rate' in df.columns:
            df['heart_rate'] = df['heart_rate'].fillna(df['heart_rate'].median())
        if 'glucose' in df.columns:
            imputer = KNNImputer(n_neighbors=10)
            df[['glucose']] = imputer.fit_transform(df[['glucose']])
    
    expected_columns = ['age', 'sex', 'cigs_per_day', 'bp_meds', 'prevalent_stroke', 
                        'prevalent_hyp', 'diabetes', 'total_cholesterol', 'bmi', 
                        'heart_rate', 'glucose', 'systolic_bp', 'diastolic_bp']
    for col in expected_columns:
        if col not in df.columns:
            df[col] = 0
    
    df = df.astype({'age': int, 'sex': int, 'cigs_per_day': int, 'bp_meds': int, 
                    'prevalent_stroke': int, 'prevalent_hyp': int, 'diabetes': int,
                    'total_cholesterol': float, 'bmi': float, 'heart_rate': float, 'glucose': float})
    
    df['pulse_pressure'] = df['systolic_bp'] - df['diastolic_bp']
    df.drop(['systolic_bp', 'diastolic_bp'], axis=1, inplace=True)
    
    continuous_var = ['age', 'cigs_per_day', 'total_cholesterol', 'bmi', 'heart_rate', 'glucose', 'pulse_pressure']
    
    for col in continuous_var:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = np.log10(df[col] + 1)
    
    imputer = SimpleImputer(strategy='median')
    for col in continuous_var:
        df[col] = imputer.fit_transform(df[[col]])
    
    
    return df

@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path, encoding='utf-8')

@st.cache_resource
def train_evaluate_model(df):
    X = df.drop(['ten_year_chd', 'education', 'id', 'is_smoking'], axis=1)
    y = df['ten_year_chd']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y, shuffle=True)
    
    smote = SMOTE()
    X_smote, y_smote = smote.fit_resample(X_train, y_train)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_smote_scaled = scaler.fit_transform(X_smote)
    X_test_scaled = scaler.transform(X_test)
    
    models = {
        "Logistic Regression": LogisticRegression(),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=7),
        "Naive Bayes": GaussianNB(var_smoothing=1e-2)
    }
    
    results = {}
    
    for model_name, model in models.items():
        model.fit(X_smote_scaled, y_smote)
        y_pred = model.predict(X_test_scaled)
        recall = recall_score(y_test, y_pred)
        results[model_name] = (model, recall)
    
    return results, scaler


def recommend_fitness_plan(age, gender, risk_status):
    fitness_plans = {
        'High': {
            'Infants': """Ensure adequate nutrition with breastfeeding or formula. Encourage tummy time and supervised play to promote motor skills development.""",
            'Children (1-4)': """Encourage active playtime, including activities like running, jumping, and climbing. Provide a balanced diet rich in fruits, vegetables, and whole grains.""",
            'Children (5-12)': """Promote at least one hour of physical activity daily. Include sports, active games, and swimming. Ensure a balanced diet and regular pediatric check-ups.""",
            'Adolescents (13-17)': """Encourage participation in team sports or individual physical activities like running, cycling, and swimming. Promote a balanced diet and regular health monitoring.""",
            'Young Adults (18-24)': """Adopt a heart-healthy diet, avoid smoking, and engage in regular physical activity including HIIT, strength training, and flexibility exercises.""",
            'Adults (25-39)': """Maintain a balanced diet, avoid smoking, and engage in regular physical activity such as jogging, cycling, and swimming. Incorporate strength training and flexibility exercises.""",
            '40-50': """Follow a strict heart-healthy diet, manage stress, and engage in moderate-intensity exercises. Include activities such as brisk walking, jogging, and resistance training. Consult a healthcare provider for a personalized exercise plan.""",
            '50-60': """Prioritize a heart-healthy diet, manage medications, and engage in low-impact exercises. Activities such as walking, water aerobics, and light strength training are beneficial. Regular consultations with a cardiologist are important.""",
            'Above 60': """Focus on a heart-healthy diet, regular medication adherence, and gentle physical activities. Activities like walking, stretching, and yoga can be helpful. Regular health check-ups and consultations with a cardiologist are essential."""
        },
        'Low': {
            'Infants': """Ensure adequate nutrition with breastfeeding or formula. Encourage tummy time and supervised play to promote motor skills development.""",
            'Children (1-4)': """Encourage active playtime, including activities like running, jumping, and climbing. Provide a balanced diet rich in fruits, vegetables, and whole grains.""",
            'Children (5-12)': """Promote at least one hour of physical activity daily. Include sports, active games, and swimming. Ensure a balanced diet and regular pediatric check-ups.""",
            'Adolescents (13-17)': """Encourage participation in team sports or individual physical activities like running, cycling, and swimming. Promote a balanced diet and regular health monitoring.""",
            'Young Adults (18-24)': """Maintain a balanced diet and engage in regular physical activity including jogging, cycling, and strength training. Incorporate flexibility exercises.""",
            'Adults (25-39)': """Maintain a balanced diet and engage in regular physical activity such as jogging, cycling, or swimming. Incorporate strength training and flexibility exercises.""",
            '40-50': """Include aerobic exercises, strength training, and flexibility exercises. Monitor your cholesterol levels regularly and maintain a healthy diet.""",
            '50-60': """Engage in moderate physical activities such as walking, swimming, and light strength training. Focus on maintaining a balanced diet and regular health check-ups.""",
            'Above 60': """Prioritize low-impact exercises such as walking, water aerobics, and gentle yoga. Ensure regular health check-ups and maintain a balanced diet. Incorporate activities to improve balance and flexibility."""
        }
    }
    
    if age < 1:
        age_group = 'Infants'
    elif 1 <= age < 5:
        age_group = 'Children (1-4)'
    elif 5 <= age < 13:
        age_group = 'Children (5-12)'
    elif 13 <= age < 18:
        age_group = 'Adolescents (13-17)'
    elif 18 <= age < 25:
        age_group = 'Young Adults (18-24)'
    elif 25 <= age < 40:
        age_group = 'Adults (25-39)'
    elif 40 <= age < 50:
        age_group = '40-50'
    elif 50 <= age <= 60:
        age_group = '50-60'
    else:
        age_group = 'Above 60'
    
    risk_category = 'High' if risk_status == "Positive" else 'Low'
    fitness_plan = fitness_plans[risk_category][age_group]
    
    return fitness_plan

# Load and preprocess data
df = load_data('C:\\Users\\zee7s\\Downloads\\cardiovascular_risk.csv')
df = preprocess_data(df)
results, scaler = train_evaluate_model(df)

# Main interface function
def streamlit_interface():
    
    st.markdown("""
        <style>
            body {
                background-color: #f0f2f6;
            }
            .stApp {
                background-color: #ffffff;
                border: 1px solid #e6e6e6;
                padding: 20px;
                border-radius: 10px;
            }
            .stButton > button {
                background-color: #4CAF50;
                color: white;
                padding: 10px 20px;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                font-size: 16px;
            }
            .stButton > button:hover {
                background-color: #45a049;
            }
            .stRadio > div {
                flex-direction: row;
                justify-content: space-evenly;
            }
            .stRadio > div > label {
                margin-right: 10px;
                font-size: 16px;
            }
            .stNumberInput > div > div {
                border-radius: 5px;
                border: 1px solid #ccc;
            }
            .stNumberInput > div > div:focus-within {
                border: 1px solid #4CAF50;
            }
            .stTextInput > div > div {
                border-radius: 5px;
                border: 1px solid #ccc;
            }
            .stTextInput > div > div:focus-within {
                border: 1px solid #4CAF50;
            }
            .stMarkdown {
                font-size: 16px;
            }
            .stTitle {
                color: #4CAF50;
                text-align: center;
            }
        </style>
    """, unsafe_allow_html=True)

    def extract_information(text):
        extracted_fields = {}

        name_pattern = r'(?:name|Name|NAME)\s*[=:]\s*([A-Za-z]+)'
        match = re.search(name_pattern, text)
        if match:
            extracted_fields['name'] = match.group(1).strip()

        
        # Extract age
        age_pattern = r'Age:\s*(\d+)'
        match = re.search(age_pattern, text)
        if match:
            extracted_fields['age'] = int(match.group(1).strip())

        # Extract gender
        gender_pattern = r'Sex:\s*(Male|Female)'
        match = re.search(gender_pattern, text)
        if match:
            extracted_fields['gender'] = match.group(1).strip()

        # Extract height
        height_pattern = r'Height:\s*(\d+)'
        match = re.search(height_pattern, text)
        if match:
            extracted_fields['height'] = int(match.group(1).strip())

        # Extract weight
        weight_pattern = r'Weight:\s*(\d+)'
        match = re.search(weight_pattern, text)
        if match:
            extracted_fields['weight'] = int(match.group(1).strip())

        # Extract smoker or non smoker
        smoking_pattern = r'Smoking:\s*(Yes|No)'
        match = re.search(smoking_pattern, text)
        if match:
            extracted_fields['smoking'] = match.group(1).strip()
        else:
            extracted_fields['smoking'] = None
        
        # Extract number of cigarettes per day if the person is a smoker
        if extracted_fields['smoking'] == 'Yes':
            cigs_per_day_pattern = r'cigarettes\s*per\s*day:\s*(\d+)'
            match = re.search(cigs_per_day_pattern, text, flags=re.IGNORECASE)
            if match:
                extracted_fields['cigs_per_day'] = int(match.group(1).strip())
            else:
                extracted_fields['cigs_per_day'] = None
        else:
            extracted_fields['cigs_per_day'] = None

        # Extract heart rate
        heart_rate_pattern = r'Heart\s*Rate:\s*(\d+)'
        match = re.search(heart_rate_pattern, text)
        if match:
            extracted_fields['heart_rate'] = int(match.group(1).strip())

        # Extract glucose
        glucose_pattern = r'Glucose:\s*(\d+(\.\d+)?)'
        match = re.search(glucose_pattern, text)
        if match:
            extracted_fields['glucose'] = float(match.group(1).strip())

        # Extract systolic BP
        systolic_bp_pattern = r'Systolic\s*BP:\s*(\d+(\.\d+)?)'
        match = re.search(systolic_bp_pattern, text)
        if match:
            extracted_fields['systolic_bp'] = float(match.group(1).strip())

        # Extract diastolic BP
        diastolic_bp_pattern = r'Diastolic\s*BP:\s*(\d+(\.\d+)?)'
        match = re.search(diastolic_bp_pattern, text)
        if match:
            extracted_fields['diastolic_bp'] = float(match.group(1).strip())

        return extracted_fields



    enter = st.selectbox("enter your information", ['enter manually', 'upload file'])

    if enter == 'enter manually':

        name = st.text_input("Name")
        age = st.number_input("Age", min_value=0, max_value=120, step=1)
        sex = st.radio("Sex", ["Male", "Female"])
        height = st.number_input("Height (cm)", min_value=50, max_value=250, step=1)
        weight = st.number_input("Weight (kg)", min_value=10, max_value=300, step=1)
        smoking = st.radio("Smoking", ["No", "Yes"])
        cigsperday = st.number_input("Cigarettes Per Day", min_value=1, max_value=100, step=1) if smoking == "Yes" else 0
        heart_rate = st.number_input("Heart Rate", min_value=30, max_value=200, step=1)
        glucose = st.number_input("Glucose", min_value=1.0)
        systolic_bp = st.number_input("Systolic BP", min_value=1.0)
        diastolic_bp = st.number_input("Diastolic BP", min_value=1.0)

        
    else:
        uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

    

        if uploaded_file is not None:
            image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)
            gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            extracted_text = pytesseract.image_to_string(gray_img)
            st.write(extracted_text)
            extracted_fields = extract_information(extracted_text)
            
        else:
            extracted_fields = {}

        st.markdown("### User Details")
        name = st.text_input("Name", value=extracted_fields.get('name', ''))
        age = st.number_input("Age", min_value=0,    max_value=120, step=1, value=extracted_fields.get('age', None))
        sex = st.radio("Sex", ["Male", "Female"], index=0 if extracted_fields.get('gender') == 'Male' else 1)
        height = st.number_input("Height (cm)", min_value=50, max_value=250, step=1, value=extracted_fields.get('height', None))
        weight = st.number_input("Weight (kg)", min_value=10, max_value=300, step=1, value=extracted_fields.get('weight', None))
        smoking = st.radio("Smoking", ["No", "Yes"], index=1 if extracted_fields.get('smoking') == 'Yes' else 0)
        cigsperday = st.number_input("Cigarettes Per Day", min_value=1, max_value=100, step=1, value=extracted_fields.get('cigs_per_day', None)) if smoking == "Yes" else 0
        heart_rate = st.number_input("Heart Rate", min_value=30, max_value=200, step=1, value=extracted_fields.get('heart_rate', None))
        glucose = st.number_input("Glucose", min_value=1.0, value=extracted_fields.get('glucose', None))
        systolic_bp = st.number_input("Systolic BP", min_value=1.0, value=extracted_fields.get('systolic_bp', None))
        diastolic_bp = st.number_input("Diastolic BP", min_value=1.0, value=extracted_fields.get('diastolic_bp', None))


    if st.button("Predict and Recommend"):
        user_data = pd.DataFrame({
            'age': [age],
            'sex': [1 if sex == 'Male' else 0],
            'cigs_per_day': [cigsperday],
            'bp_meds': [0],
            'prevalent_stroke': [0],
            'prevalent_hyp': [0],
            'diabetes': [0],
            'total_cholesterol': [200],
            'bmi': [weight / ((height/100) ** 2)],
            'heart_rate': [heart_rate],
            'glucose': [glucose],
            'systolic_bp': [systolic_bp],
            'diastolic_bp': [diastolic_bp]
        })

        user_data = preprocess_data(user_data, is_user_input=True)
        user_data = user_data[['age', 'sex', 'cigs_per_day', 'bp_meds', 'prevalent_stroke', 'prevalent_hyp', 
                            'diabetes', 'total_cholesterol', 'bmi', 'heart_rate', 'glucose', 'pulse_pressure']]
        user_data_scaled = scaler.transform(user_data)

        risk = None
        for model_name, (model, model_recall) in results.items():
            y_pred = model.predict(user_data_scaled)
            if y_pred[0] == 1:
                risk = "Positive"
                break
            else:
                risk = "Negative"

        st.markdown("### User Details")
        with st.expander("View Details"):
            st.markdown(f"Name: {name}")
            st.markdown(f"Age: {age}")
            st.markdown(f"Sex: {'Male' if sex == 'Male' else 'Female'}")
            st.markdown(f"Height: {height} cm")
            st.markdown(f"Weight: {weight} kg")
            st.markdown(f"Smoking: {smoking}")
            st.markdown(f"Cigarettes Per Day: {cigsperday}")
            st.markdown(f"Heart Rate: {heart_rate}")
            st.markdown(f"Glucose: {glucose}")
            st.markdown(f"Systolic BP: {systolic_bp}")
            st.markdown(f"Diastolic BP: {diastolic_bp}")

        st.markdown("### Prediction Result")
        st.markdown(f"Cardiovascular Disease Risk: {risk}")

        fitness_plan = recommend_fitness_plan(age, sex, risk)
        st.markdown("### Fitness Plan Recommendation")
        st.markdown(fitness_plan)
def app():
    st.markdown("""
        <style>
            body {
                background-color: #f0f2f6;
            }
            .stApp {
                background-color: #ffffff;
                border: 1px solid #e6e6e6;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.1);
            }
            .stTitle, .stHeader {
                color: #4CAF50;
                font-family: 'Arial', sans-serif;
            }
            .stTextInput > div > input {
                background-color: #ffffff;
                border: 1px solid #e6e6e6;
                border-radius: 5px;
                padding: 10px;
                font-size: 16px;
            }
            .stRadio > div {
                padding: 5px 0;
            }
            .stNumberInput input {
                border-radius: 5px;
                border: 1px solid #e6e6e6;
                padding: 10px;
                font-size: 16px;
            }
            .stButton > button {
                background-color: #4CAF50;
                color: white;
                padding: 10px 20px;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                font-size: 16px;
            }
            .stButton > button:hover {
                background-color: #45a049;
            }
        </style>
    """, unsafe_allow_html=True)
def app():
    st.markdown("""
        <style>
            body {
                background-color: #f0f2f6;
            }
            .stApp {
                background-color: #ffffff;
                border: 1px solid #e6e6e6;
                padding: 20px;
                border-radius: 10px;
            }
            .stTitle, .stHeader {
                color: #4CAF50;
                font-family: 'Arial', sans-serif;
            }
            .stTextInput > div > input {
                background-color: #ffffff;
                border: 1px solid #e6e6e6;
                border-radius: 5px;
                padding: 10px;
                font-size: 16px;
            }
            .stButton > button {
                background-color: #4CAF50;
                color: white;
                padding: 10px 20px;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                font-size: 16px;
            }
            .stButton > button:hover {
                background-color: #45a049;
            }
            .stSelectbox > div {
                border: 1px solid #e6e6e6;
                border-radius: 5px;
                padding: 10px;
            }
            .stAlert {
                border-radius: 5px;
                padding: 10px;
                margin: 10px 0;
            }
        </style>
    """, unsafe_allow_html=True)    
    
    st.title('Cardiovascular Disease Prediction System')
    
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False

    if st.session_state.logged_in:
        streamlit_interface()
    else:
        choice = st.selectbox("Login/Signup", ['Login', 'Signup'])

        if choice == 'Login':
            email = st.text_input("Enter Email here")
            password = st.text_input("Enter password here", type='password')
            
            if st.button('Login'):
                try:
                    user = auth.get_user_by_email(email)
                    st.session_state.logged_in = True
                    st.experimental_rerun()
                except:
                    st.warning("Login failed!")

        else:
            email = st.text_input("Enter Email here")
            password = st.text_input("Enter password here", type='password')
            username = st.text_input("Enter username here")
            
            if st.button('Create account'):
                try:
                    user = auth.create_user(email=email, password=password, uid=username)
                    st.success("Account created! Please login.")
                except Exception as e:
                    st.error(f"Account creation failed: {e}")

app()