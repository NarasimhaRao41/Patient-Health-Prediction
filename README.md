# 🏥 Patient Health Prediction System

A Machine Learning–based system that predicts patient health risk using clinical and lifestyle data.  
This project demonstrates an end-to-end ML workflow including data preprocessing, model training, evaluation, and deployment using a web application.

---

## 📌 Project Overview

Early prediction of patient health risks (such as cardiovascular disease) can help in timely medical intervention.  
This project uses supervised machine learning techniques to analyze patient records and predict health outcomes.

The system allows users to:
- Enter patient details
- Get a **risk prediction**
- View an **explanation** of the prediction

---

## 🧠 Machine Learning Workflow

1. **Data Collection**
   - Dataset: `CVD_cleaned.csv`
   - Contains patient demographic, lifestyle, and medical features

2. **Data Preprocessing**
   - Handling missing values
   - Encoding categorical features
   - Feature scaling
   - Train–test split

3. **Model Training**
   - Algorithms used:
     - Logistic Regression
     - Random Forest
     - Support Vector Machine (SVM)
   - Best performing model saved as a `.pkl` file

4. **Model Evaluation**
   - Accuracy
   - Precision
   - Recall
   - F1-score
   - Confusion Matrix

5. **Deployment**
   - Flask-based web application (`app.py`)
   - User-friendly interface
   - Prediction explanation shown to users

---

## 🗂️ Project Structure

Patient-Health-Prediction/
│
├── app.py # Flask web application
├── train_model.py # Model training script
├── CVD_cleaned.csv # Dataset
├── cvd_model.pkl # Trained ML model
├── feature_names.pkl # Feature list used during training
├── .gitignore # Git ignore file
└── README.md # Project documentation

yaml
Copy code

---

## 🚀 How to Run the Project

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/NarasimhaRao41/Patient-Health-Prediction.git
cd Patient-Health-Prediction
2️⃣ Create Virtual Environment (Optional but Recommended)
bash
Copy code
python -m venv venv
venv\Scripts\activate   # For Windows
3️⃣ Install Dependencies
bash
Copy code
pip install -r requirements.txt
(If requirements.txt is not present, install manually)

bash
Copy code
pip install flask pandas numpy scikit-learn
4️⃣ Train the Model
bash
Copy code
python train_model.py
5️⃣ Run the Web App
bash
Copy code
python app.py
Open browser and go to:

cpp
Copy code
http://127.0.0.1:5000/
📊 Technologies Used
Python

Flask

Pandas

NumPy

Scikit-learn

Matplotlib / Seaborn

HTML & CSS

📈 Future Enhancements
Add SHAP or LIME for better explainability

Deploy using Streamlit / Render / AWS

Add user authentication

Improve UI/UX

Use deep learning models

👨‍💻 Author
Narasimha Rao
GitHub: NarasimhaRao41

