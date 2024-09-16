# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
### STEP 1:
Import the required packages and print the present data.
### STEP 2:
Find the null and duplicate values.

### STEP 3:
Using logistic regression find the predicted values of accuracy , confusion matrices.

### STEP 4:
Display the results.


## Program:
```
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Naveenaa A K
RegisterNumber:  212222230094
```


```
import pandas as pd 
data=pd.read_csv("C:/Users/admin/Desktop/INTR MACH/Placement_Data.csv")
data.head()

data1=data.copy() 
data1=data1.drop(["sl_no" , "salary"] , axis=1)
data1.head()

data1.isnull().sum() 

data1.duplicated().sum()


from sklearn.preprocessing import LabelEncoder
le= LabelEncoder()
data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])
data1

x=data1.iloc[: , :-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test=train_test_split(x,y, test_size=0.2, random_state=0)

from sklearn. linear_model import LogisticRegression 
lr= LogisticRegression (solver = "liblinear") #library for Large Linear classification 1r.fit(x_train,y_train)
lr.fit(x_train , y_train)
y_pred =lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test , y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion= (y_test , y_pred)
confusion


lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
  
```

## Output:
![image](https://github.com/user-attachments/assets/f45b5451-3b45-4689-a513-acc8747d80f4)
![image](https://github.com/user-attachments/assets/04054cea-8da1-460b-8a13-58e7d8c4775f)
![image](https://github.com/user-attachments/assets/c4b9ac52-6c27-49ca-afba-22169af2b0e0)
![image](https://github.com/user-attachments/assets/c7fd7337-f79a-4bd4-bae7-f3768e5336e7)
![image](https://github.com/user-attachments/assets/8d237597-44c7-4735-b365-0e7eddeb8a8e)
![image](https://github.com/user-attachments/assets/5a9f18a9-78ae-4b60-bc8d-fb7bab1050dc)
![image](https://github.com/user-attachments/assets/56f68b7f-92be-42e4-8813-ca2f3bc6cdb4)
![image](https://github.com/user-attachments/assets/ec6cac2a-6ffe-464b-a18d-dadc44344fcc)
![image](https://github.com/user-attachments/assets/79f5caff-cec2-4a75-b8fe-e7c111e4bd63)
![image](https://github.com/user-attachments/assets/909bfb14-6ee8-4cf3-b738-2568e4e21076)
![image](https://github.com/user-attachments/assets/16e803f2-2fc8-46fb-989f-1832d3526c62)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
