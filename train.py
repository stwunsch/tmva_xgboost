# First XGBoost model for Pima Indians dataset
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# load data
dataset = np.loadtxt('pima-indians-diabetes.data.csv', delimiter=",")

# split data into X and y
num_features = 2 # max is 8
X = dataset[:,0:num_features]
Y = dataset[:,8]
print(X)
print(Y)

# split data into train and test sets
seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

# fit model no training data
model = XGBClassifier(n_estimators=2)
model.fit(X_train, y_train)

# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]

# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

model.get_booster().dump_model("model.json", dump_format="json")

for idx in range(20):
    in_ = X[idx]
    out_ = model.predict_proba(np.array([X[idx]]))
    print("{0:<16} -> {1:16}".format(in_.squeeze(), out_.squeeze()))
