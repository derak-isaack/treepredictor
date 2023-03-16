import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib as joblib
import ravel as ravel


tree_df = pd.read_csv("export_dataframes.csv")
tree_df.columns=["Height" ,"LongestBrach", "TrunkDiameter","LeafThickness","Trees"]

print(tree_df)
tree_df.sample(frac=1, random_state=5)

x = tree_df[["Height","LongestBrach","TrunkDiameter","LeafThickness"]]
y = tree_df[["Trees"]]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=5, stratify = y)

clf = RandomForestClassifier( n_estimators = 100)

clf.fit(x_train, y_train.values.ravel())

y_pred = clf.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"accuracy:{accuracy}")


joblib.dump(clf, "Untitled22.sav")







