import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier
# from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
import warnings

warnings.filterwarnings("ignore")

# ------------------ Load Dataset ------------------
# Ensure 'diabetes.csv' is in the same directory
data = pd.read_csv("diabetes.csv")

# Split into features and labels
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ------------------ Define Classifiers ------------------
classifiers = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    # "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    "Extra Trees": ExtraTreesClassifier(n_estimators=100, random_state=42),
    "Decision Tree": DecisionTreeClassifier(),
    "Logistic Regression": LogisticRegression(),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
    "ANN": MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000)
}

# ------------------ Train & Evaluate ------------------
results = {}
for clf_name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    print(f"{clf_name}: {score * 100:.2f}%")
    results[clf_name] = score

# ------------------ Best Model ------------------
best_model_name = max(results, key=results.get)
best_model = classifiers[best_model_name]

print(f"\nBest Classifier: {best_model_name} with accuracy {results[best_model_name]*100:.2f}%")

# ------------------ Custom Test Inputs ------------------

# Example test cases (manually entered as in your demo)
# Format: [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]

#x_test = [[6, 148, 72, 35, 0, 33.6, 0.627, 50]]  # Expected: Diabetic (1)
#x_test = [[1, 85, 66, 29, 0, 26.6, 0.351, 31]]   # Expected: Non-diabetic (0)
x_test = [[5, 100, 70, 20, 85, 29.5, 0.5, 28]]   # Custom example
# x_test = [[2, 120, 65, 30, 100, 32.0, 0.2, 35]]  # Custom example

prediction = best_model.predict(x_test)[0]

print("\nPrediction for custom input:")
print("Diabetes Detected ✅" if prediction == 1 else "No Diabetes ❌")

# ==============================================================================================================================================

OUTPUT :

Random Forest: 72.08%
Extra Trees: 74.03%
Decision Tree: 75.32%
Logistic Regression: 74.68%
KNN: 66.23%
Naive Bayes: 76.62%
ANN: 64.94%

Best Classifier: Naive Bayes with accuracy 76.62%

Prediction for custom input:
No Diabetes ❌
