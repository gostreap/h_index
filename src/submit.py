from read_data import get_test_data
from preprocess_utils import get_submission_data, normalize
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

def submit(model):
    X_train, y_train, X_test, y_test = get_submission_data()
    # X_train, X_test = normalize(X_train, X_test)

    print(X_train.shape)
    print(X_test.shape)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    test, _ = get_test_data()
    test["hindex"] = y_pred
    submission = test[["author", "hindex"]]
    submission.to_csv("../tmp/submission.csv", index=None)

def submit_mean():
    model_lgbm = LGBMRegressor(n_estimators=4000)
    model_cat = CatBoostRegressor(verbose=False, task_type="GPU", iterations=40000)

    X_train, y_train, X_test, y_test = get_submission_data()

    print(X_train.shape, X_test.shape)

    model_cat.fit(X_train, y_train)
    cat_pred = model_cat.predict(X_test)

    model_lgbm.fit(X_train, y_train)
    lgbm_pred = model_lgbm.predict(X_test)

    y_pred = (lgbm_pred + cat_pred) / 2

    test, _ = get_test_data()
    test["hindex"] = y_pred
    submission = test[["author", "hindex"]]
    submission.to_csv("../tmp/submission.csv", index=None)