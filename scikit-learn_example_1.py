### The code below was designed to train a multiple linear regression model to predict the correct tooling to use for a proprietary manufacturing process.  As such, the dataset is not included as it contained confidential information.

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import f_regression
from sklearn.metrics import mean_absolute_error
df = pd.read_csv(r"C:\Users\Data_Input.csv")
df2 = df.drop(columns=(["Order", "ID Tool"]))
X = df2.drop(columns=(["OD Tool", "Speed", "Model Wall"]))
y = df2["OD Tool"]
feature_names = []
for z in X.columns:
    feature_names.append(z)
p_value_array = f_regression(X=X, y=y)
p_values = pd.DataFrame(data=p_value_array, columns=feature_names)
K_Fold_10 = np.random.randint(1,500,10)
for q in K_Fold_10:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=z)
    pipe = make_pipeline(StandardScaler(), LinearRegression())
    pipe_ft = pipe.fit(X=X_train, y=y_train)
    coefs = list(zip(pipe_ft.named_steps['linearregression'].coef_, feature_names))
    coefs_df = pd.DataFrame(data=coefs, columns=("Coef", "Variable"))
    intercept = pipe_ft.named_steps['linearregression'].intercept_
    df2 = pd.DataFrame(data= {"Coef":[intercept], "Variable":["Intercept"]})
coefs_all = coefs_df.append(df2)
coefs_all.to_csv(f"C:/Users/Output_csv_files/{q}",index=False)

