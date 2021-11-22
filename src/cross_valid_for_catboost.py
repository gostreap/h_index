from catboost import Pool, cv

def cross_valid(X,y,Kfold = 3):
    
    cv_data = Pool(data = X, label=y)
    params = {
        "loss_function":"RMSE",
    }
    score  = cv(cv_data,params,fold_count =3, plot = "True")[1]
    return score
