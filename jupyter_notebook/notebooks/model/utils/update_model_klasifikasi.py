import sys
import os

current_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(current_dir)
import preprocessing as pr

# Add 'work/commons' to sys.path
commons_path = os.path.abspath(os.path.join(current_dir, "../../commons"))
sys.path.append(commons_path)
import g_lib as gl

import pandas as pd
import logging

class CustomError(Exception):
    def __init__(self, error_dict):
        self.error_dict = error_dict

    def __str__(self):
        return repr(self.error_dict)

import pandas as pd
import numpy as np
import copy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import f1_score
import joblib

def create_model_param():
    """Create the model objects"""

    xgb_params = {
        "n_estimators": [100],
        "learning_rate": [0.3],
        "max_depth": [3, 10],
        "min_child_weight": [1, 10],
        "gamma": [0, 5],
    }

    # rf_params = {
    #     "n_est  # imators": [100],
    #     "criterion": ["entropy"],
    #     "random_state": [42],
    # }

    # Create model params
    list_of_param = {"XGBClassifier": xgb_params, 
                   # "RandomForestClassifier": rf_params
                     }

    return list_of_param


def create_model_object():
    """Create the model objects"""
    print("Creating model objects")

    # Create model objects
    xgb = XGBClassifier()
    # rf = RandomForestClassifier()

    # Create list of model
    list_of_model = [
        {"model_name": xgb.__class__.__name__, "model_object": xgb},
        # {"model_name": rf.__class__.__name__, "model_object": rf},
    ]

    return list_of_model
    
def train_model(data, column_name, pred_column_name):
    """Function to get the best model"""
    # Convert tokenized texts to string format
    X = [" ".join(tokens) for tokens in data[column_name]]

    # Encode labels into numerical values
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(data[pred_column_name])

    # Split train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Oversampling Data Training
    print("Before OverSampling, counts of each class:")
    for i in range(np.max(y_train) + 1):
        print("Label {}: {}".format(i, sum(y_train == i)))

    # Create TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer()

    # Transform the training data into TF-IDF features
    X_train = tfidf_vectorizer.fit_transform(X_train)

    # Transform the testing data using the same vectorizer
    X_test = tfidf_vectorizer.transform(X_test)

    sm = SMOTE(random_state=42)
    X_train, y_train = sm.fit_resample(X_train, y_train)

    print("After OverSampling, counts of each class:")
    for i in range(np.max(y_train) + 1):
        print("Label {}: {}".format(i, sum(y_train == i)))

    # Create list of params & models
    list_of_param = create_model_param()
    list_of_model = create_model_object()

    # List of trained model
    list_of_tuned_model = []

    # Train model
    for base_model in list_of_model:
        # Current condition
        model_name = base_model["model_name"]
        model_obj = copy.deepcopy(base_model["model_object"])
        model_param = list_of_param[model_name]

        # Debug message
        print("Training model :", model_name)

        # Create model object
        model = RandomizedSearchCV(
            estimator=model_obj,
            param_distributions=model_param,
            n_iter=3,
            cv=3,
            random_state=42,
            n_jobs=1,
            verbose=10,
            scoring={"f1_macro": "f1_macro", "accuracy": "accuracy"},
            refit="f1_macro",
        )

        # Train model
        model.fit(X_train, y_train)

        # Get the best estimator with best hyperparameters
        best_estimator = model.best_estimator_

        # Predict
        y_pred_train = best_estimator.predict(X_train)
        y_pred_test = best_estimator.predict(X_test)

        # Get F1-score (macro average)
        train_score = f1_score(y_train, y_pred_train, average="macro")
        test_score = f1_score(y_test, y_pred_test, average="macro")

        list_of_tuned_model.append(
            {
                "model_name": model_name,
                "model": best_estimator,
                "train_auc": train_score,
                "test_auc": test_score,
                "best_params": model.best_params_,
            }
        )

        print("Done training")
        print("")

    return {
            "list_of_param": list_of_param,
            "list_of_model": list_of_model,
            "list_of_tuned_model": list_of_tuned_model,
            "tfidf_vectorizer": tfidf_vectorizer,
            "label_encoder": label_encoder,
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
        }


def get_best_model(list_of_tuned_model):
    """Function to get the best model"""

    # Get the best model
    best_model_name = None
    best_model = None
    best_performance = -99999
    best_model_param = None

    for model_entry in list_of_tuned_model:
        model_name = model_entry["model_name"]
        test_auc = model_entry["test_auc"]
        if test_auc > best_performance:
            best_model_name = model_name
            best_model = model_entry["model"]
            best_performance = test_auc
            best_model_param = model_entry["best_params"]

    # Print
    print("=============================================")
    print("Best model        :", best_model_name)
    print("Metric score      :", best_performance)
    print("Best model params :", best_model_param)
    print("=============================================")

    return {
            "best_model": best_model,
            "model_name": best_model_name,
            "metric_score": best_performance,
            "model_params": best_model_param,
        }


def get_best_threshold(X_test, y_test, best_model):
    """Function to tune & get the best decision threshold"""

    # Get the proba pred
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]

    # Initialize
    metric_threshold = pd.Series([])

    THRESHOLD = np.linspace(0, 1, 100)
    # Optimize
    for threshold_value in THRESHOLD:
        # Get predictions
        y_pred = (y_pred_proba >= threshold_value).astype(int)

        # Get the F1 score
        metric_score = f1_score(y_test, y_pred, average="macro")

        # Add to the storage
        metric_threshold[metric_score] = threshold_value

    # Find the threshold @max metric score
    metric_score_max_index = metric_threshold.index.max()
    best_threshold = metric_threshold[metric_score_max_index]
    print("=============================================")
    print("Best threshold :", best_threshold)
    print("Metric score   :", metric_score_max_index)
    print("=============================================")

    return {
            "best_threshold": best_threshold,
            "metric_score": metric_score_max_index,
        }


def start_batching():
    data = gl.get_worksheet_to_df(file_name='Data Training ML', sheet_name='review_classification')
    
    data = pr.run_tokenizer(data, 'review_text')
    
    # Train & Optimize the model
    tm = train_model(data, 'review_text','topic')

    # Get the best model
    list_of_tuned_model = tm["list_of_tuned_model"]
    bm = get_best_model(list_of_tuned_model)

    # Get the best threshold for the best model
    X_test = tm["X_test"]
    y_test = tm["y_test"]
    bt = get_best_threshold(X_test, y_test, bm["best_model"])

    # # Get Model Log
    # model_record = {
    #     "timestamp": [utils.time_stamp()],
    #     "model_name": [bm["model_name"]],
    #     "model_params": [bm["model_params"]],
    #     "threshold": [bt["best_threshold"]],
    #     "test_accuracy_score": [bm["metric_score"]],
    #     "test_metric_score": [bt["metric_score"]],
    # }

    # data_model_record = utils.load_json(CONFIG_DATA["bm_record_t"])
    # data_model_record = pd.concat(
    #     [data_model_record, pd.DataFrame(model_record)], axis=0, ignore_index=True
    # )

    # Running Dump File
    # utils.dump_json(data, CONFIG_DATA["data_set_path_t"])
    # utils.dump_json(all_data_clean, CONFIG_DATA["data_clean_path_t"])
    joblib.dump(tm["tfidf_vectorizer"], 'tfidf_vectorizer.pkl')
    joblib.dump(tm["label_encoder"], 'label_encoder.pkl')

    # Best Model
    # if (
    #     data_model_record["test_accuracy_score"].iloc[-1]
    #     >= data_model_record["test_accuracy_score"].max()
    # ):
    #     utils.pickle_dump(bm["best_model"], 'model_klasifikasi.pkl')
    #     utils.pickle_dump(bt["best_threshold"], 'best_threshold.pkl')
    #     utils.dump_json(data_model_record, CONFIG_DATA["bm_record_t"])
    #     print("Metric score increased, new model updated")
    #     response = {'status': 100, 'message': 'Success with model updated'}
    #     print(response)
    #     return response
    # else:
    #     utils.dump_json(data_model_record, CONFIG_DATA["bm_record_t"])
    #     print("Metric score not increased, new model not updated")
    #     response = {'status': 200, 'message': 'Success with model not updated'}
    #     print(response)
    #     return response
    joblib.dump(bm["best_model"], 'model_klasifikasi.pkl')

if __name__ == "__main__":
    
    try:
        logging.info('starting')
        start_batching()

    except Exception as e:
        print("An error occurred:", str(e))
        response = {'status': 300, 'message': 'Error Script'}
        raise CustomError(response)
