import pickle

import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

if __name__ == '__main__':
    # Read the iris dataset
    iris_data = pd.read_csv('../../resources/iris.csv')

    # Separate features and target
    X = iris_data.iloc[:, :-1]  # All rows, all columns except last
    y = iris_data.iloc[:, -1]  # All rows, last column

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    # Create and train the SVM Classifier
    svm_clf = SVC(kernel='rbf', random_state=42)
    svm_clf.fit(X_train, y_train)

    # Make predictions
    # y_pred = svm_clf.predict(X_test)

    # Print model performance metrics
    # print(f"Confusion Matrix:")
    # print(confusion_matrix(y_test, y_pred))
    # print("\nAccuracy Score:", accuracy_score(y_test, y_pred))
    # print("\nClassification Report:")
    # print(classification_report(y_test, y_pred))

    # Save the model to a file
    with open('../../resources/models/model_cls_iris_v1.0.pkl', 'wb') as file:
        pickle.dump(svm_clf, file)
        print("Model saved successfully.")