### The function below was designed to train an (intentionally) overfit classification tree.
### This was done to automate a very manual process (one that a truly predictive decision tree utilizing a pruned validation set
###    could not replicate due to a lack of consistency in the base data).
### It utilizes a for-loop to iterate over multiple files to perform k-fold cross validation,  
###    extract the base trained tree containing decisions & terminal node information,
###    and then create and export multiple .csv files  to a specific folder for use elsewhere.
### The data used to train the model is confidential, and thus, unavailable for example purposes.

def BMG():
    import numpy as np
    import pandas as pd
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.tree import export_text
    import os
    Input_files = os.listdir("C:/Users/Input_csv_files")
    for file in Input_files:
        z = file
        df = pd.read_csv(f"C:/Users/Input_csv_files/{z}")
        df2 = df.drop(columns=(["OD Wall", "Count"]))
        X = df2.drop(columns=df2.columns[-1])
        y = df2.iloc[:,-1]
        X1 = pd.get_dummies(data=X, columns=["Grade Group"])
        feature_names = []
        for x in X1.columns:
            feature_names.append(x)
        tree_model = DecisionTreeRegressor(random_state=1)
        trained_tree = tree_model.fit(X=X1, y=y)
        tree_text = export_text(decision_tree=trained_tree, feature_names=feature_names, decimals=9, max_depth=500)
        tree_in_prog = tree_text.splitlines()
        tree_df = pd.DataFrame(data=tree_in_prog).rename(columns={0:"Rule"})
        tree_df["Node_ID"] = range(1, 1+len(tree_df))
        tree_df["Node_Type_Mask"] = tree_df["Rule"].str.endswith("]")
        tree_df["Node_if_TRUE"] = tree_df["Node_ID"] + 1
        tree_df.loc[tree_df["Node_Type_Mask"] == True, "Node_if_TRUE"] = 0
        tree_df["Bar_Count"] = tree_df["Rule"].str.count("\\|")
        tree_df["Split_Variable"] = "NA"
        for x in feature_names:
            tree_df.loc[tree_df["Rule"].str.contains(pat=x, regex=False), "Split_Variable"] = x
        tree_df["Operator"] = "NA"
        operator_list = [">","<="]
        for x in operator_list:
            tree_df.loc[tree_df["Rule"].str.contains(pat=x, regex=False), "Operator"] = x
        tree_df["Value"] = "NA"
        tree_df.loc[tree_df["Node_Type_Mask"] == False, "Value"] = tree_df["Rule"].str[-11:]
        temp_df = (tree_df["Rule"]
           .str.split(pat="[", expand=True)
           .drop(columns=0)
           .squeeze()
           .str.split(pat="]", expand=True)
           .drop(columns=1)
           .rename(columns={0:"T_Value"})
          )
        tree_df.loc[tree_df["Node_Type_Mask"] == True, "Value"] = temp_df["T_Value"]
        final_tree = tree_df.drop(columns="Rule")
        z = z.replace("Data", "Tree").replace("Input", "Output")
        final_tree.to_csv(f"C:/Users/Output_csv_files/{z}",index=False)