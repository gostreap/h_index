from catboost import CatBoostRegressor, Pool
from preprocess_utils import get_numpy_data, get_processed_data, get_submission_data, select_columns, TRAIN_LENGTH, add_features
from read_data import get_test_data
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
import pandas as pd

data = get_processed_data(split=False)
data = select_columns(data)

columns = [column for column in data if column.startswith("tf")]
data = data.drop(columns, axis=1)

n_features = 5000

full_data = data[['author','text']]
data = full_data[-full_data.text.isna()]
vectorizer = TfidfVectorizer(max_features = n_features)
X = vectorizer.fit_transform(data.text.values)
tfid = pd.DataFrame(X.toarray(),columns=["tf"+str(i) for i in range(n_features)])
tfid.index = full_data[-full_data.text.isna()].index
datavf = pd.concat([full_data[['author']],tfid], axis=1)

r = SimpleImputer(strategy='mean').fit_transform(datavf[["tf"+str(i) for i in range(n_features)]])
datavf[["tf"+str(i) for i in range(n_features)]] = r

data = data.add_features(data, datavf)

data = data.drop("author", axis=1)
train = data[:TRAIN_LENGTH]
test = data[TRAIN_LENGTH:]

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
