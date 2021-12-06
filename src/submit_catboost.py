from catboost import CatBoostRegressor, Pool
from preprocess_utils import get_numpy_data, get_processed_data, get_submission_data, select_columns
from read_data import get_test_data
from sklearn.model_selection import train_test_split

train, test = get_processed_data()
train = select_columns(train)
test = select_columns(test)

print(train.columns)

train_split, test_split = train_test_split(train)

X_train = train_split.drop(["author", "hindex"], axis=1)
y_train = train_split["hindex"]
X_test = test_split.drop(["author", "hindex"], axis=1)
y_test = test_split["hindex"]

print(X_train.shape)

train_pool = Pool(
    data=X_train,
    label=y_train
)

valid_pool = Pool(data=X_test, label=y_test)
model_cat = CatBoostRegressor(
    random_state=1,
    iterations=100000,
    task_type="GPU",
    depth=8
)
model_cat.fit(train_pool, eval_set=valid_pool, use_best_model=True)

X_train = train.drop(["author", "hindex"], axis=1)
y_train = train["hindex"]
X_test = test.drop(["author", "hindex"], axis=1)
y_test = test["hindex"]

train_pool = Pool(
    data=X_train,
    label=y_train,
)

model_cat.fit(train_pool)

y_pred = model_cat.predict(X_test)

test, _ = get_test_data()
test["hindex"] = y_pred
submission = test[["author", "hindex"]]
submission.to_csv("../tmp/submission.csv", index=None)
