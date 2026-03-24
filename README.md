#  Credit Card Customer Churn Prediction

##  Project Overview

This project predicts whether a customer will **leave (churn)** or **stay** using a machine learning model built with a Neural Network.

Churn prediction helps businesses:

* Identify risky customers
* Take action to retain them
* Improve revenue

---

##  Dataset

* Dataset used: **BankChurners.csv**
* Target column: `Attrition_Flag`

  * `0` → Existing Customer
  * `1` → Attrited Customer (Churn)

---

##  Technologies Used

* Python
* Pandas (data handling)
* Scikit-learn (preprocessing & evaluation)
* TensorFlow / Keras (model building)
* Pickle (saving scaler)

---

##  Step-by-Step Workflow

### 1️. Load Dataset

```python
df = pd.read_csv("data/BankChurners.csv")
```

* Reads the dataset into a DataFrame.

---

### 2️. Data Cleaning

```python
df = df.drop([...], axis=1)
```

* Removes unnecessary columns (like IDs and redundant features).

---

### 3️. Convert Target Variable

```python
df['Attrition_Flag'] = df['Attrition_Flag'].map({
    "Existing Customer": 0,
    "Attrited Customer": 1
})
```

* Converts text labels into numeric values.

---

### 4️. Convert Categorical Data

```python
df = pd.get_dummies(df, drop_first=True)
```

* Converts categorical columns into numeric format.

---

### 5️. Split Features & Target

```python
X = df.drop('Attrition_Flag', axis=1)
y = df["Attrition_Flag"]
```

---

### 6️. Train-Test Split

```python
train_test_split(X, y, test_size=0.2, random_state=42)
```

* 80% → Training
* 20% → Testing

---

### 7️. Feature Scaling

```python
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

* Normalizes data for better model performance.

---

### 8️. Save Scaler

```python
pickle.dump(scaler, open("Scaler.pkl", "wb"))
```

* Saves scaling logic for future predictions.

---

### 9️. Build Neural Network

```python
model = Sequential()
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
```

👉 Explanation:

* Input Layer → 16 neurons
* Hidden Layer → 8 neurons
* Output Layer → 1 neuron (binary classification)

---

### 10. Compile Model

```python
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
```

* **Adam** → Optimizer (updates weights)
* **Binary Crossentropy** → Loss function for classification
* **Accuracy** → Performance metric

---

### 1️1️. Train Model

```python
model.fit(X_train, y_train, epochs=50, batch_size=32)
```

* Learns patterns from training data.

---

### 1️2️. Evaluate Model

```python
model.evaluate(X_test, y_test)
```

* Checks performance on unseen data.

---

### 1️3️. Predictions

```python
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.3)
```

👉 Threshold = **0.3**

* Converts probability → 0 or 1
* Lower threshold = catches more churn customers

---

### 1️4️. Model Evaluation

```python
confusion_matrix(y_test, y_pred)
classification_report(y_test, y_pred)
```

* Confusion Matrix → shows correct/incorrect predictions
* Classification Report → precision, recall, f1-score

---

### 1️5️. Save Model

```python
model.save("model.h5")
```

* Saves trained model for future use

---

##  Model Performance

* Accuracy: ~92%
* Recall (Churn): High (important for business)
* Threshold used: **0.3**

---

##  How to Run the Project

### Step 1: Clone Repository

```bash
git clone <your-repo-link>
cd credit_card_churn_prediction
```

### Step 2: Create Virtual Environment

```bash
python -m venv .venv
.venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run the Model

```bash
python model/train.py
```

---

##  Project Structure

```
credit_card_churn_prediction/
│
├── data/
│   └── BankChurners.csv
│
├── model/
│   └── train.py
│
├── model.h5
├── scaler.pkl
├── README.md
```

---

##  Key Learnings

* Importance of preprocessing
* Neural network basics
* Threshold tuning
* Model evaluation techniques

---

##  Future Improvements

* Handle class imbalance
* Hyperparameter tuning
* Use ROC-AUC metric
* Deploy using Flask/Streamlit

---

##  Author

* B.GAYATHRI

---

⭐ If you like this project, consider giving it a star!
