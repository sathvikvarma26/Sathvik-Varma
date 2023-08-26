#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
d1=pd.read_excel(r"Lab Session1 Data (1).xlsx",sheet_name='Purchase data')

d1.drop(d1.iloc[:,5:22],inplace=True,axis=1)
A=d1.iloc[:,1:-1].values
C=d1.iloc[:,-1].values
A=np.array(A)
C=np.array(C)

print("Matrix of A:")
print(A)

print("Matrix of C:")
print(C)

rank=np.linalg.matrix_rank(A)
print("rank of Matrix A:", rank)

inverse=np.linalg.pinv(A)
print("inverse of A: ",inverse)

Pseudo_inv=np.matmul(inverse,C)
print("Pseudo inverse is actual cost of each product  : ",Pseudo_inv)

t=np.array(d1['Payment (Rs)'])
number=len(t)
New_cat=[]
for i in range(0,number):
    if t[i]>200:
        New_cat.append("rich")
    else:
        New_cat.append("poor")
d1.insert(loc = 5,column = 'Label',value = New_cat)
print("New Data Excel Sheet for Purchase Data : ")
print(d1)

X = d1.drop(['Customer', 'Payment (Rs)', 'Label'], axis=1) 
y = d1['Label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 
scaler = StandardScaler() # Feature scaling
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled) 
print(classification_report(y_test, y_pred))


# In[ ]:




