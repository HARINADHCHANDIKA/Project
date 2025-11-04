import os
from flask import Flask, render_template, request, redirect, url_for, flash
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import warnings

warnings.filterwarnings("ignore")

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

# Global variables
model = None
scaler = None
le_country = None
le_gender = None
X_columns = None  # to preserve column order for model input

def safe_currency_format(value):
    try:
        val = float(value)
        return f"₹{val:,.2f}"
    except:
        return value  # already formatted or invalid

def train_and_save_model():
    global model, scaler, le_country, le_gender, X_columns

    try:
        dataset_path = r'C:\Users\USER\Desktop\bank_churn_app\bank_churn_app\Bank Customer Churn Prediction.csv'
        df = pd.read_csv(dataset_path)

        df.drop(columns=['customer_id'], inplace=True)

        # Add "India" samples for country example
        india_samples = df.sample(100, replace=True).copy()
        india_samples['country'] = 'India'
        df = pd.concat([df, india_samples], ignore_index=True)

        # Handle outliers
        numeric_cols = df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            if col == 'churn':
                continue
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            upper = q3 + 1.5 * iqr
            lower = q1 - 1.5 * iqr
            df[col] = np.where(df[col] > upper, upper,
                        np.where(df[col] < lower, lower, df[col]))

        # Encoding
        le_country = LabelEncoder()
        le_gender = LabelEncoder()
        df['country'] = le_country.fit_transform(df['country'])
        df['gender'] = le_gender.fit_transform(df['gender'])

        # Split
        X = df.drop('churn', axis=1)
        y = df['churn']
        X_columns = X.columns  # Save for later input alignment

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Scale
        scaler = StandardScaler()
        columns_to_scale = ['credit_score', 'age', 'tenure', 'balance', 
                            'products_number', 'estimated_salary']
        X_train[columns_to_scale] = scaler.fit_transform(X_train[columns_to_scale])
        X_test[columns_to_scale] = scaler.transform(X_test[columns_to_scale])

        # Train model
        model = XGBClassifier(
            n_estimators=200, 
            learning_rate=0.05, 
            max_depth=6, 
            random_state=42,
            eval_metric='logloss'
        )
        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)
        print("Model Accuracy:", accuracy_score(y_test, y_pred))
        print(classification_report(y_test, y_pred))

        # Save model and preprocessors
        with open('model.pkl', 'wb') as f:
            pickle.dump(model, f)
        with open('preprocessor.pkl', 'wb') as f:
            pickle.dump({
                'scaler': scaler,
                'le_country': le_country,
                'le_gender': le_gender,
                'columns': list(X.columns)
            }, f)
        print("Model and preprocessing saved successfully.")

    except Exception as e:
        print(f"Training Error: {str(e)}")
        raise

def load_model_and_preprocessor():
    global model, scaler, le_country, le_gender, X_columns

    try:
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('preprocessor.pkl', 'rb') as f:
            preprocessor = pickle.load(f)
            scaler = preprocessor['scaler']
            le_country = preprocessor['le_country']
            le_gender = preprocessor['le_gender']
            X_columns = preprocessor['columns']
        print("Model and preprocessor loaded.")
        return True

    except Exception as e:
        print(f"Load Error: {str(e)}. Retraining model...")
        train_and_save_model()
        return load_model_and_preprocessor()

# Load model when app starts
print("Initializing...")
load_model_and_preprocessor()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            customer_data = {
                'credit_score': float(request.form['credit_score']),
                'country': request.form['country'],
                'gender': request.form['gender'],
                'age': float(request.form['age']),
                'tenure': float(request.form['tenure']),
                'balance': float(request.form['balance']),
                'products_number': float(request.form['products_number']),
                'credit_card': 1 if request.form.get('credit_card') == 'on' else 0,
                'active_member': 1 if request.form.get('active_member') == 'on' else 0,
                'estimated_salary': float(request.form['estimated_salary'])
            }

            input_data = pd.DataFrame([customer_data])

            # Encode country
            try:
                input_data['country'] = le_country.transform([customer_data['country']])[0]
            except ValueError:
                input_data['country'] = 0
                flash(f"Note: Country '{customer_data['country']}' not recognized. Using default.", 'info')

            # Encode gender
            try:
                input_data['gender'] = le_gender.transform([customer_data['gender']])[0]
            except ValueError:
                input_data['gender'] = 0
                flash(f"Note: Gender '{customer_data['gender']}' not recognized. Using default.", 'info')

            # Scale
            columns_to_scale = ['credit_score', 'age', 'tenure', 'balance', 
                                'products_number', 'estimated_salary']
            input_data[columns_to_scale] = scaler.transform(input_data[columns_to_scale])

            # Ensure column order
            input_data = input_data[X_columns]

            # Predict
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0][1]

            # Format values safely
            customer_data['balance'] = safe_currency_format(customer_data['balance'])
            customer_data['estimated_salary'] = safe_currency_format(customer_data['estimated_salary'])

            status = "ACTIVE" if prediction == 0 else "INACTIVE (Churn Risk)"
            suggestions = generate_suggestions(customer_data, prediction)

            return render_template('results.html',
                                   status=status,
                                   probability=f"{probability * 100:.2f}%",
                                   customer_data=customer_data,
                                   suggestions=suggestions)

        except Exception as e:
            flash(f"Prediction Error: {str(e)}", "danger")
            return redirect(url_for('predict'))

    return render_template('predict.html')

def generate_suggestions(customer_data, prediction):
    suggestions = []

    # Handle balance formatting safely
    balance_str = str(customer_data['balance'])
    if '₹' in balance_str:
        balance = float(balance_str.replace("₹", "").replace(",", ""))
    else:
        balance = float(balance_str)

    age = float(customer_data['age'])
    products = float(customer_data['products_number'])

    if prediction == 1:  # churn risk
        suggestions.append("We value your business! Here's how we can serve you better:")

        if balance > 50000:
            suggestions.append("💰 Consider our High Net Worth program with exclusive benefits")
        elif balance < 1000:
            suggestions.append("💸 Our zero-balance account might suit you better")

        if products < 2:
            suggestions.append("📊 Explore our additional financial products")
            if age < 35:
                suggestions.append("🎓 Check out youth investment plans with low minimums")

        if customer_data['credit_card'] == 0:
            suggestions.append("💳 Apply for our RuPay credit card with 5% cashback on UPI")

        if age > 55:
            suggestions.append("👵 Senior citizen benefits: Higher FD rates & priority services")
        elif age < 25:
            suggestions.append("👩‍🎓 Student benefits: Zero charges & education loan discounts")

        suggestions.append("🎁 Limited offer: 0.5% extra interest on savings for 6 months")
        suggestions.append("📱 Download our app for instant loan approvals & quick support")

    else:
        suggestions.append("Thank you for banking with us! Did you know:")
        if products < 3:
            suggestions.append("🔍 You may qualify for premium services with your balance")
        if customer_data['active_member'] == 0:
            suggestions.append("⚡ Activate your account for higher interest rates")

    if customer_data.get('country') == 'India':
        suggestions.append("🇮🇳 Special for Indian customers: Free NEFT/RTGS transactions")
        suggestions.append("🪙 10% bonus reward points on UPI transactions this month")

    return suggestions

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
