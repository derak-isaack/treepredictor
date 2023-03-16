import joblib

def predict(data):
    clf = joblib.load("Untitled22.sav")
    result = clf.predict(data)
    return clf.predict(data)