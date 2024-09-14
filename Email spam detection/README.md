# Spam Detection with Logistic Regression

This project demonstrates a basic spam detection system using logistic regression. The system classifies messages as either spam or ham (not spam) based on their content.

## Requirements

- Python 3.x
- NumPy
- Pandas
- Scikit-learn

## Installation

Install the required libraries using pip:

```bash
pip install numpy pandas scikit-learn
```

## Dataset

The dataset used is `mail_data.csv`, which contains two columns:
- `Category`: Indicates whether the message is spam (0) or ham (1).
- `Message`: The text content of the message.

## Steps

1. **Load the Data**:
    ```python
    import pandas as pd
    df = pd.read_csv('mail_data.csv')
    data = df.where(pd.notnull(df), '')
    data.loc[data['Category'] == 'spam', 'Category'] = 0
    data.loc[data['Category'] == 'ham', 'Category'] = 1
    ```

2. **Prepare the Data**:
    ```python
    X = data['Message']
    Y = data['Category']
    from sklearn.model_selection import train_test_split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)
    ```

3. **Feature Extraction**:
    ```python
    from sklearn.feature_extraction.text import TfidfVectorizer
    feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
    X_train_feature = feature_extraction.fit_transform(X_train)
    X_test_feature = feature_extraction.transform(X_test)
    Y_train = Y_train.astype('int')
    Y_test = Y_test.astype('int')
    ```

4. **Train the Model**:
    ```python
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression()
    model.fit(X_train_feature, Y_train)
    ```

5. **Evaluate the Model**:
    ```python
    from sklearn.metrics import accuracy_score
    Y_predict_train = model.predict(X_train_feature)
    accuracy_train = accuracy_score(Y_train, Y_predict_train)
    print(f"Training Accuracy: {accuracy_train}")

    Y_predict_test = model.predict(X_test_feature)
    accuracy_test = accuracy_score(Y_test, Y_predict_test)
    print(f"Test Accuracy: {accuracy_test}")
    ```

6. **Predict New Messages**:
    ```python
    inputt = ["Your message here"]
    inputt_feature = feature_extraction.transform(inputt)
    inputt_predict = model.predict(inputt_feature)
    print(inputt_predict)
    ```

## Example

```python
inputt = ["Dear Valued Customer, You have won a $1,000 gift card!"]
inputt_feature = feature_extraction.transform(inputt)
inputt_predict = model.predict(inputt_feature)
print(inputt_predict)  # Output: [0] (spam)

inputt2 = ["Hi Team, This is a reminder about our meeting."]
inputt_feature2 = feature_extraction.transform(inputt2)
inputt_predict2 = model.predict(inputt_feature2)
print(inputt_predict2)  # Output: [1] (ham)
```

## Conclusion

This project provides a simple example of how to build a spam detection system using logistic regression and TF-IDF vectorization.