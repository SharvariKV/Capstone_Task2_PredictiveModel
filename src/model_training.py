import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder,OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from imblearn.over_sampling import SMOTE

#Self-contained data loading

def load_and_prep_data():
    possible_paths=[
        "data/cleaned_data.csv",
        "data/raw_data.csv",
        "data/train.csv"
    ]

    data_path=None
    for path in possible_paths:
        if os.path.exists(path):
            data_path=path
            break

    if data_path is None:
        raise FileExistsError(
            "Could not find data in 'data/' folder."
            "Please ensure raw_data.csv, train.csv, or cleaned_data.csv exists inside data/."
        ) 

    print(f"Loading data from: {data_path}")
    df=pd.read_csv(data_path)

    if 'Embarked' in df.columns:
        df['Embarked']=df['Embarked'].fillna('S')
    if 'Fare' in df.columns:
        df['Fare']=df['Fare'].fillna(df['Fare'].median())

    #Save cleaned reference copy inside data/
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/cleaned.csv",index=False)

    X = df[['Pclass', 'Sex', 'Fare', 'Embarked']].copy()
    y = df['Survived'].copy()
    return X, y

#Main modeling pipeline execution
def main():
    X, y=load_and_prep_data()

    # 2. Train / Test Split BEFORE fitting encoders or scalers
    X_train,X_test,y_train,y_test=train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # 3. Setup Preprocessing ColumnTransformer
    ordinal_features=['Pclass']
    pclass_order=[[1, 2, 3]] #3rd class lowest 1st class highest
    categorical_features=['Sex', 'Embarked']
    numeric_features=['Fare']

    preprocessor = ColumnTransformer(
        transformers = [
            ('num', StandardScaler(), numeric_features),
            ('ord', OrdinalEncoder(categories=pclass_order), ordinal_features),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
        ]
    )

    #Check class imbalance ratio
    minority_ratio = (y_train.sum()/len(y_train))*100
    print(f"\nMinority Class (Survived=1) Percentage in Train Set: {minority_ratio:.2f}%")

    #Prepare Transformed sets for Model Baseline vs. SMOTE Evaluation
    X_train_prep = preprocessor.fit_transform(X_train)
    X_test_prep = preprocessor.fit_transform(X_test)

    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_prep, y_train)


    models= {
     'Logistic Regression': LogisticRegression(random_state=42),
     'Decision Tree': DecisionTreeClassifier(random_state=42),
     'Random Forest': RandomForestClassifier(random_state=42)
    }

    results=[]
    print("\n-----Evaluating 3 Models-----")
    for name,model in models.items():
       #Metrics before SMOTE
       model.fit(X_train_prep, y_train)
       y_pred_before = model.predict(X_test_prep)
       f1_before = f1_score(y_test, y_pred_before, pos_label=1)

       #Metrics after SMOTE
       model.fit(X_train_res, y_train_res)
       y_pred_after = model.predict(X_test_prep)
       acc = accuracy_score(y_test, y_pred_after)
       prec = precision_score(y_test, y_pred_after, pos_label=1)
       rec = recall_score(y_test, y_pred_after, pos_label=1)
       f1_after = f1_score(y_test, y_pred_after, pos_label=1)
       cm = confusion_matrix(y_test, y_pred_after)

       results.append({
         'Model':name,
         'Accuracy':round(acc,4),
         'Precision':round(prec,4),
         'Recall':round(rec,4),
         'F1(Before SMOTE)':round(f1_before,4),
         'Binary F1 (Primary)': round(f1_after,4)
        })

       print(f"\n---{name}---")
       print(f"Accuracy:{acc:.4f} | Precision:{prec:.4f} | Recall:{rec:.4f} | Binary F1:{f1_after:.4f}")
       print("Confusion Matrix:\n",cm)

    #Print Comparison table
    results_df = pd.DataFrame(results)
    print("\n----Model Cmparison Table----")
    print(results_df.to_string(index=False))

    full_pipeline = Pipeline(steps=[
        ('preprocessor',preprocessor),
        ('classifier', LogisticRegression(random_state=42))
    ])

    #5-Fold stratified cross validation
    cv_results = cross_validate(
        full_pipeline,X_train, y_train, cv=5, scoring='f1', return_train_score=False
    )

    print("\n---5-Fold Stratified cv results(Logistic Regression)---")
    print(f"Fold Scores: {np.round(cv_results['test_score'],4)}")
    print(f"Mean F1 Score: {cv_results['test_score'].mean():.4f}")
    print(f"Standard Deviation: {cv_results['test_score'].std():.4f}")

    #Hyperparameter Grid Search
    param_grid={
        'classifier__C': [0.01,0.1,1.0,10.0],
        'classifier__solver': ['liblinear', 'lbfgs']
    }

    grid_search = GridSearchCV(
        estimator=full_pipeline,
        param_grid=param_grid,
        cv=5,
        scoring='f1',
        n_jobs=-1
    )

    grid_search.fit(X_train,y_train)

    print(("\n---Hyperparameter Search Results---"))
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Best Validation F1-Score:{grid_search.best_score_:.4f}")

    #Final Evaluation on Held-Out Test Set
    best_tuned_model = grid_search.best_estimator_
    y_final_pred = best_tuned_model.predict(X_test)
    final_f1 = f1_score(y_test, y_final_pred, pos_label=1)

    print(f"Final Tuned Test Set F1-Score:{final_f1:.4f}")

    #Save final model
    os.makedirs("models",exist_ok=True)
    joblib.dump(best_tuned_model, "models/trained_model.pkl")
    print("\nSaved best model artifact to models/trained_model.pkl")


if __name__ == "__main__":
    main()
    