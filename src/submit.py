from read_data import get_train_data, get_test_data
from fasttext_utils import get_submission_data, normalize, small_class, df_to_txt, format_data
from utils import get_core_number, get_min_coauthor_hindex, get_max_coauthor_hindex, get_page_rank
import pandas as pd
import fasttext

def submit(model):
    X_train, y_train, X_test, y_test = get_submission_data()
    X_train, X_test = normalize(X_train, X_test)

    print(X_train.shape)
    print(X_test.shape)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    test, _ = get_test_data()
    test["hindex"] = y_pred
    submission = test[["author", "hindex"]]
    submission.to_csv("../tmp/submission.csv", index=None)