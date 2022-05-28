import pandas as pd
import numpy as np

data = pd.read_excel("C:/Users/shivarajmb/Desktop/projects/project1/sampla_data_08_05_2022(final).xlsx")
data.columns

data1= data.drop(["Patient_ID"], axis=1)
data1= data1.drop(["Patient_Age"], axis=1)
data1= data1.drop(["Patient_Gender"], axis=1)
data1= data1.drop(["Test_Booking_Date"], axis=1)
data1= data1.drop(["Sample_Collection_Date"], axis=1)
data1= data1.drop(["Mode_Of_Transport"], axis=1)
data1.columns

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data2=data1.copy()
data2["Test_Name"] = le.fit_transform(data1["Test_Name"])
data2["Sample"] = le.fit_transform(data2["Sample"])
data2["Way_Of_Storage_Of_Sample"] = le.fit_transform(data2["Way_Of_Storage_Of_Sample"])
data2["Cut-off Schedule"] = le.fit_transform(data2["Cut-off Schedule"])
data2["Traffic_Conditions"] = le.fit_transform(data2["Traffic_Conditions"])
data2["Reached_On_Time"] = le.fit_transform(data2["Reached_On_Time"])

# Input and Output Split
predictors = data2.loc[:, data2.columns!='Reached_On_Time']
type(predictors)

target = data2['Reached_On_Time']
type(target)

# Train Test partition of the data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(predictors, target, test_size = 0.3, random_state=2)


########################## Adaboost ##########################################

from sklearn.ensemble import AdaBoostClassifier

ada_clf = AdaBoostClassifier(learning_rate = 0.05, n_estimators = 500)

ada_clf.fit(x_train, y_train)

from sklearn.metrics import accuracy_score, confusion_matrix

# Evaluation on Testing Data
confusion_matrix(y_test, ada_clf.predict(x_test))
accuracy_score(y_test, ada_clf.predict(x_test))

# Evaluation on Training Data
accuracy_score(y_train, ada_clf.predict(x_train))

final_df = pd.concat([predictors,target], axis=1)

input_data = (6,0,1,16.1,10.15,0,13.15,4,0,12,72,3,9,54)

input_data_as_numpy_array = np.asarray(input_data)

input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = ada_clf.predict(input_data_reshaped)
print(prediction)

if prediction == 1:
    print("Reached on time")
else:
    print("Wont reach on time")


import pickle
filename = "train.pkl"
pickle.dump(ada_clf, open(filename,"wb"))

filename1 = "final.pkl"
pickle.dump(data1, open(filename1,"wb"))
