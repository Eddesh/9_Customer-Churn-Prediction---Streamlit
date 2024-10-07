import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
from sklearn.preprocessing import OneHotEncoder
import warnings
warnings.filterwarnings("ignore")

class DataProcessor:
    def __init__(self, filename, target):
        self.df = pd.read_csv(filename)
        self.target = target

    def clean_column_names(self):
        self.df.columns = [col.replace(' ', '_').replace('.', '_').lower() for col in self.df.columns]

    def drop_columns(self, columns_to_drop):
        self.df = self.df.drop(columns_to_drop, axis=1)

    def fill_missing_values(self, column_name, strategy):
        if strategy == 'median':
            fill_value = self.df[column_name].median()
        elif strategy == 'mean':
            fill_value = self.df[column_name].mean()
        elif strategy == 'mode':
            fill_value = self.df[column_name].mode()[0]
        else:
            raise ValueError("Strategy must be 'median', 'mean', or 'mode'")
        self.df[column_name].fillna(fill_value, inplace=True)

    def split_data(self, test_size=0.2, random_state=42):
        X = self.df.drop(self.target, axis=1)
        y = self.df[self.target]
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

class FeatureEngineering:
    def __init__(self, encodings, OHE_cols, scale_cols):
        self.encodings = encodings
        self.OHE_cols = OHE_cols
        self.scale_cols = scale_cols
        self.scaler = StandardScaler()

    def binary_encode(self, train_data, test_data):
        for col, encoding in self.encodings.items():
            train_data[col] = train_data[col].replace(encoding)
            test_data[col] = test_data[col].replace(encoding)
        return train_data, test_data

    def one_hot_encode(self, train_data, test_data, OHE_encoder):
        encoded_train = OHE_encoder.fit_transform(train_data[self.OHE_cols])
        encoded_train_df = pd.DataFrame(encoded_train.toarray(), columns=OHE_encoder.get_feature_names_out())
        train_data = train_data.reset_index()
        train_data = pd.concat([train_data.drop(self.OHE_cols, axis=1), encoded_train_df], axis=1)
        
        encoded_test = OHE_encoder.transform(test_data[self.OHE_cols])
        encoded_test_df = pd.DataFrame(encoded_test.toarray(), columns=OHE_encoder.get_feature_names_out())
        test_data = test_data.reset_index()
        test_data = pd.concat([test_data.drop(self.OHE_cols, axis=1), encoded_test_df], axis=1)
        
        return train_data, test_data

    def scaling(self, train_data, test_data):
        train_data[self.scale_cols] = self.scaler.fit_transform(train_data[self.scale_cols])
        test_data[self.scale_cols] = self.scaler.transform(test_data[self.scale_cols])
        return train_data, test_data

class ModelTrainer:
    def __init__(self, parameters, model_type):
        self.parameters = parameters
        self.model_type = model_type
        self.grid_search = GridSearchCV(self.model_type, param_grid=self.parameters, scoring='accuracy', cv=5)

    def with_grid_search(self, x_train, y_train):
        self.grid_search.fit(x_train, y_train)
        return self.grid_search.best_params_

    def train_with_best_params(self, x_train, y_train, best_params):
        model = XGBClassifier(**best_params, random_state=42)
        model.fit(x_train, y_train)
        return model

class ModelEvaluator:
    def __init__(self, model):
        self.model = model

    def evaluate_classification_report(self, x_test, y_test, target_names):
        y_pred = self.model.predict(x_test)
        return classification_report(y_test, y_pred, target_names=target_names)

    def evaluate_metrics(self, x_test, y_test):
        y_pred = self.model.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='binary')
        recall = recall_score(y_test, y_pred, average='binary')
        f1 = f1_score(y_test, y_pred, average='binary')
        return accuracy, precision, recall, f1

def main():
    data_processor = DataProcessor("data_C.csv", "churn")
    data_processor.clean_column_names()

    drop_columns = ['unnamed:_0', 'id', 'customerid', 'surname']
    data_processor.drop_columns(drop_columns)
    data_processor.fill_missing_values('creditscore', 'median')
    x_train, x_test, y_train, y_test = data_processor.split_data()

    encodings = {"gender": {"Male": 0, "Female": 1}}
    OHE_cols = ['geography']
    scale_cols = ['creditscore', 'balance', 'estimatedsalary']
    feature_engineer = FeatureEngineering(encodings, OHE_cols, scale_cols)

    x_train, x_test = feature_engineer.binary_encode(x_train, x_test)
    x_train, x_test = feature_engineer.one_hot_encode(x_train, x_test, OneHotEncoder())
    x_train, x_test = feature_engineer.scaling(x_train, x_test)

    # Model training
    parameters = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.1, 0.01, 0.001],
        'subsample': [0.5, 0.7, 1]
    }
    model_trainer = ModelTrainer(parameters, XGBClassifier())
    best_params = model_trainer.with_grid_search(x_train, y_train)
    model = model_trainer.train_with_best_params(x_train, y_train, best_params)

    # Model evaluation
    model_evaluator = ModelEvaluator(model)
    report = model_evaluator.evaluate_classification_report(x_test, y_test, ['Not Churn', 'Churn'])
    accuracy, precision, recall, f1 = model_evaluator.evaluate_metrics(x_test, y_test)

    # Print results
    print("Classification Report:")
    print(report)
    print("Evaluation Metrics:")
    print(f"Accuracy: {accuracy}\nPrecision: {precision}\nRecall: {recall}\nF1 Score: {f1}\n\n")

if __name__ == "__main__":
    main()