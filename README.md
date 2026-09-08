❤️ Heart Disease Prediction System Using Machine Learning

📌 Project Overview

The Heart Disease Prediction System is a Machine Learning-based project that predicts whether a person is likely to have heart disease based on various medical and health-related attributes.

The project uses Supervised Machine Learning classification algorithms to learn patterns from historical patient data and make predictions for new patient records.

The main objective of this project is to demonstrate how Machine Learning can be applied to healthcare data for early risk prediction and decision support.

«Disclaimer: This project is intended for educational and research purposes only. It is not a substitute for professional medical diagnosis or treatment.»

---


🎯 Objectives

- Analyze a heart disease dataset.
- Perform data preprocessing and cleaning.
- Explore relationships between medical attributes.
- Visualize important patterns in the dataset.
- Train Machine Learning classification models.
- Evaluate model performance.
- Predict the possibility of heart disease for a new patient.
- Build a simple and understandable ML-based prediction system.

---


🧠 Machine Learning Workflow

                 ┌─────────────────────┐
                 │   Heart Disease     │
                 │      Dataset        │
                 └──────────┬──────────┘
                            ↓
                 ┌─────────────────────┐
                 │ Data Preprocessing  │
                 └──────────┬──────────┘
                            ↓
                 ┌─────────────────────┐
                 │ Exploratory Data    │
                 │      Analysis       │
                 └──────────┬──────────┘
                            ↓
                 ┌─────────────────────┐
                 │ Feature Selection   │
                 └──────────┬──────────┘
                            ↓
                 ┌─────────────────────┐
                 │ Train/Test Split    │
                 └──────────┬──────────┘
                            ↓
                 ┌─────────────────────┐
                 │ ML Model Training   │
                 └──────────┬──────────┘
                            ↓
                 ┌─────────────────────┐
                 │ Model Evaluation    │
                 └──────────┬──────────┘
                            ↓
                 ┌─────────────────────┐
                 │ Heart Disease       │
                 │    Prediction       │
                 └─────────────────────┘

---


📊 Dataset

The project uses a heart disease dataset containing medical information about patients.

Typical attributes include:

Feature| Description
"age"| Age of the patient
"sex"| Gender of the patient
"cp"| Chest pain type
"trestbps"| Resting blood pressure
"chol"| Serum cholesterol
"fbs"| Fasting blood sugar
"restecg"| Resting ECG results
"thalach"| Maximum heart rate achieved
"exang"| Exercise-induced angina
"oldpeak"| ST depression induced by exercise
"slope"| Slope of peak exercise ST segment
"ca"| Number of major vessels
"thal"| Thalassemia
"target"| Heart disease presence/absence

The exact columns depend on the dataset used in the implementation.

---


🛠️ Technologies Used

Programming Language

- Python

Libraries

- NumPy – Numerical computations
- Pandas – Data manipulation and analysis
- Matplotlib – Data visualization
- Seaborn – Statistical visualization
- Scikit-learn – Machine Learning algorithms and evaluation

---


🔄 Project Methodology

1. Data Collection

The heart disease dataset is loaded into the Python environment using Pandas.

import pandas as pd

data = pd.read_csv("heart_disease_data.csv")

print(data.head())

---

2. Data Analysis

The dataset is analyzed to understand:

- Number of records
- Number of features
- Data types
- Missing values
- Statistical properties
- Distribution of target classes

Example:

data.shape
data.info()
data.describe()
data.isnull().sum()

---


3. Exploratory Data Analysis

Data visualization is performed to identify relationships between different medical attributes and heart disease.

Examples of visualizations:

- Correlation heatmap
- Distribution plots
- Count plots
- Age distribution
- Gender vs heart disease
- Chest pain type vs heart disease

Example:

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10,8))
sns.heatmap(data.corr(), annot=True)
plt.show()

---


4. Data Preprocessing

The data is prepared before training the Machine Learning model.

Typical preprocessing steps include:

- Handling missing values
- Separating features and target
- Encoding categorical variables where required
- Feature scaling where required
- Splitting the dataset into training and testing sets

X = data.drop(columns="target")
Y = data["target"]

---


5. Train-Test Split

The dataset is divided into training and testing data.

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=2,
    stratify=Y
)

The training data is used to learn patterns, while the testing data is used to evaluate how well the model performs on unseen data.

---


🤖 Machine Learning Model

The project can use a classification algorithm such as Logistic Regression to predict the target class.

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(X_train, Y_train)

After training, predictions can be generated using:

X_test_prediction = model.predict(X_test)

---


📈 Model Evaluation

The trained model is evaluated using appropriate classification metrics.

Accuracy

from sklearn.metrics import accuracy_score

test_accuracy = accuracy_score(Y_test, X_test_prediction)

print("Accuracy:", test_accuracy)

Other useful evaluation metrics include:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

Example:

from sklearn.metrics import classification_report

print(classification_report(Y_test, X_test_prediction))

---


🧪 Prediction for a New Patient

After training the model, it can be used to predict the result for a new patient's medical information.

Example:

input_data = (
    63, 1, 3, 145, 233, 1, 0,
    150, 0, 2.3, 0, 0, 1
)

import numpy as np

input_data_as_numpy_array = np.asarray(input_data)

input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

prediction = model.predict(input_data_reshaped)

if prediction[0] == 1:
    print("The person is likely to have heart disease")
else:
    print("The person is unlikely to have heart disease")

---


📁 Project Structure

Heart-Disease-Prediction/
│
├── dataset/
│   └── heart_disease_data.csv
│
├── notebooks/
│   └── heart_disease_prediction.ipynb
│
├── src/
│   └── heart_disease_prediction.py
│
├── model/
│   └── heart_disease_model.pkl
│
├── requirements.txt
│
└── README.md

If your project is only a Jupyter Notebook, you can simplify the structure to:

Heart-Disease-Prediction/
│
├── heart_disease_prediction.ipynb
├── heart_disease_data.csv
├── requirements.txt
└── README.md

---

📦 Installation

1. Clone the Repository

git clone https://github.com/your-username/heart-disease-prediction.git

2. Navigate to the Project

cd heart-disease-prediction

3. Install Required Libraries

pip install -r requirements.txt

4. Run the Project

For Jupyter Notebook:

jupyter notebook

Open:

heart_disease_prediction.ipynb

---

📋 Requirements

Create a "requirements.txt" file:

numpy
pandas
matplotlib
seaborn
scikit-learn
jupyter

---

💡 Example Output

Enter patient information...

Prediction:
The person is likely to have heart disease.

or

Prediction:
The person is unlikely to have heart disease.

---

🔍 Key Features

- Machine Learning-based prediction
- Medical dataset analysis
- Data preprocessing
- Exploratory Data Analysis
- Data visualization
- Classification model
- Model accuracy evaluation
- Prediction on new patient data
- Simple and beginner-friendly implementation

---

🚀 Future Enhancements

The project can be improved by adding:

- Multiple Machine Learning algorithms
- Hyperparameter tuning
- Cross-validation
- Feature selection
- ROC-AUC analysis
- Interactive web interface
- Flask/Streamlit deployment
- Model explainability using SHAP
- Patient prediction history
- Cloud deployment

Modern heart-disease ML systems increasingly compare multiple classifiers and use metrics beyond accuracy, such as ROC-AUC, precision, recall and F1-score.

---

⚠️ Limitations

- The prediction depends on the quality and representativeness of the training dataset.
- Model accuracy does not guarantee correct diagnosis for an individual.
- Medical datasets may contain demographic and sampling biases.
- The system should not be used as an independent medical diagnostic tool.

Research implementations demonstrate that ML can be useful for cardiovascular-risk prediction, but clinical deployment requires appropriate validation and medical oversight.

---

🎓 Project Applications

This project can be used as:

- Machine Learning academic project
- Mini project
- Final-year project foundation
- Healthcare AI demonstration
- Classification algorithm demonstration
- Data Science portfolio project

---

📚 Concepts Demonstrated

This project demonstrates the following concepts:

Python
   ↓
Data Collection
   ↓
Data Preprocessing
   ↓
Exploratory Data Analysis
   ↓
Data Visualization
   ↓
Feature Selection
   ↓
Train/Test Split
   ↓
Machine Learning
   ↓
Model Evaluation
   ↓
Prediction

---

👨‍💻 Author

Your Name

- GitHub: "https://github.com/your-username"
- LinkedIn: "https://linkedin.com/in/your-profile"

---

⭐ Acknowledgement

This project is developed for educational purposes to demonstrate the application of Machine Learning techniques to heart disease prediction.

If you find this project useful, consider giving the repository a ⭐.

---

📄 License

This project is intended for educational purposes. Dataset licensing and attribution should follow the terms of the original dataset source.
