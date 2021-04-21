def classify(X, y):
    
    numeric_features = [col for col in X.columns if X[col].dtype in ['int64', 'float64']]
    categorical_features = [col for col in X.columns if X[col].dtype == "object"]
    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    grid={"classifier__C":[0,1,2,3], "classifier__penalty":["none","l1","l2"], "classifier__solver":["lbfgs"]}# l1 lasso l2 ridge

    clf = Pipeline(steps=[('preprocessor', preprocessor),('classifier', LogisticRegression(penalty="none"))])
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)
#     logreg_cv=GridSearchCV(clf,grid,cv=10,scoring='f1_micro')
#     logreg_cv.fit(X_train,y_train)
#     print("tuned hpyerparameters :(best parameters) ",logreg_cv.best_params_)
#     print("best score :",logreg_cv.best_score_)
#     print("estimator :",logreg_cv.best_estimator_)
#     print("scorer :",logreg_cv.scorer_)
#     display(logreg_cv.cv_results_)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)
    #kfold cross validation use random n_splits dataset (n_splits between 5 and 10)
    kf = KFold(10)
    scores = []
    for train_index,test_index in kf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        clf.fit(X_train, y_train)
        display(clf.score(X_train, y_train))
        display(clf.score(X_test, y_test))
        scores.append(clf.score(X_test, y_test))
    print("accuracy", np.mean(scores))
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)
    clf.fit(X_train, y_train)
    #score is mean accuracy
    display(clf.score(X_train, y_train))
    display(clf.score(X_test, y_test))
    y_pred = clf.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    conf_ma = [[tp, tn], [fp,fn]]
    df_conf_ma = pd.DataFrame(conf_ma, index=["Sucess", "Error"], columns=["Positive", "Nagative"])
    display(df_conf_ma)
    print("\tMAE: %1.3f" % mean_absolute_error(y_test, y_pred))
    print("\tPrecision: %1.3f" % precision_score(y_test, y_pred))
    print("\tRecall: %1.3f" % recall_score(y_test, y_pred))
    print("\tF1: %1.3f\n" % f1_score(y_test, y_pred))