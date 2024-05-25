from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# Load the dataset
df = pd.read_csv('StudentsPerformance.csv')
df['average score'] = df[['math score', 'reading score', 'writing score']].mean(axis=1)

# Factorize categorical columns
categorical_columns = ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']
df_categorical = df[categorical_columns]
df_categorical_encoded = df_categorical.apply(lambda x: pd.factorize(x)[0])

# Concatenate numerical and encoded categorical columns
df_encoded = pd.concat([df_categorical_encoded, df[['math score', 'reading score', 'writing score', 'average score']]], axis=1)

X = df_encoded.drop(['average score'], axis=1)
y = df_encoded['average score']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the linear regression model
lr = LinearRegression()
model = lr.fit(X_train, y_train)

# Define a route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define a route for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    data = request.form.to_dict()

    # Factorize categorical input
    input_features_categorical = pd.DataFrame([data], columns=categorical_columns).apply(lambda x: pd.factorize(x)[0])
    
    # Concatenate numerical and encoded categorical columns
    input_features = pd.concat([input_features_categorical, pd.DataFrame([data], columns=['math score', 'reading score', 'writing score'])], axis=1)

    # Make a prediction using the trained model
    prediction = model.predict(input_features)[0]

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
