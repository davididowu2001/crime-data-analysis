import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

# Load the data
def load_data():
    data = pd.read_csv("c:/Users/idowu/Desktop/Machine Learning/crime_ML/crime-analysis/Crime_Data_from_2020_to_Present.csv")
    return data

def clean_data(data):
    data = data.dropna()
    return data

# Split the data
def split_data(data):
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    return train_data, test_data


def main():
    # Load the data
    data = load_data()
    print("data loaded")

    # select the columns
    data = data[["Date Rptd", "Crm Cd", "Crm Cd Desc"]]

    data = clean_data(data)
    print("data cleaned")
    # Split the data
    train_data, test_data = split_data(data)

    # label encoding
    le = LabelEncoder()
    train_data['Crm Cd'] = le.fit_transform(train_data['Crm Cd'])
    test_data['Crm Cd'] = le.transform(test_data['Crm Cd'])
    print("data split")

    # Define pipeline
    pipeline = Pipeline([
        ("vectorizer", CountVectorizer(stop_words="english")),
        ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    with mlflow.start_run():
        # fit the model
        pipeline.fit(train_data["Crm Cd Desc"], train_data["Crm Cd"])
        # predict
        predictions = pipeline.predict(test_data["Crm Cd Desc"])
        print("predictions made")

        #add predictions to test data with inverse transform
        test_data['predictions'] = le.inverse_transform(predictions)

        #log parameters
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("random_state", 42)
        print("parameters logged")

        #log metrics
        accuracy = accuracy_score(test_data['Crm Cd'], predictions)
        print(test_data)
        print("accuracy: ", accuracy)

        mlflow.log_metric("accuracy", accuracy)

        mlflow.set_tag("model", "RandomForestClassifier to predict crime code from description")
        
        #log model
        mlflow.sklearn.log_model(pipeline, 'Random_Classifier_model')
        print("Model saved in run %s" % mlflow.active_run().info.run_uuid)

if __name__ == "__main__":
    main()