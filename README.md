# DataMining_CBS3007
DataMining-CBS3007

# Q1:-Collect the data set consists of 50 observations about patient enrolment in diet maintenance based on gender, weight, BMI etc (minimum 7 features). Implement a model that will recommend a strict diet is necessary or not for a patient using the naïve Bayes classification algorithm. (50x7)
# Aim:
To collect a dataset consisting of 50 observations about patient enrollment in diet maintenance, based on seven features such as gender, weight, BMI, exercise frequency, age, dietary habits, and medical history. The goal is to implement a Naive Bayes classification algorithm to recommend whether a strict diet is necessary for a patient based on these features.
# Sample Dataset:
Patient_ID	Gender	Age	Weight_kg	Height_cm	BMI	Enrolled_in_Diet_Plan	Exercise_Frequency_per_Week
1	Other	26	69.3	151.5	20.8	No	1
2	Male	43	65.9	154.8	28.4	Yes	1
3	Other	19	46.6	166.3	34.4	No	1
4	Other	37	58.1	152.5	27	No	0
5	Male	45	84	184.5	21.9	No	0

# Code:
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


df = pd.read_csv(r"C:\Users\adity\Documents\datamining\patient_enrolment_data.csv")


df['Strict_Diet_Necessary'] = np.where((df['BMI'] >= 30) & (df['Exercise_Frequency_per_Week'] <= 2), 'Yes', 'No')


X = df[['BMI', 'Exercise_Frequency_per_Week']]
y = df['Strict_Diet_Necessary']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = GaussianNB()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
classification_report_str = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Classification Report:\n", classification_report_str)
print("Confusion Matrix:\n", conf_matrix)
# Output:
 ![image](https://github.com/user-attachments/assets/62c2e8c3-c661-4d43-9631-6d7de978beb3)

# Result:
The Naive Bayes classification model was successfully implemented, and the accuracy of the model in predicting whether a strict diet is necessary was evaluated. The model performs well and provides useful recommendations based on features like BMI, weight, exercise frequency, and medical history.

# Q2:- Implement K-means method of clustering. Use the patient details data set to classify into 3 clusters such as a person is normal, healthy and weak. A person/patient must be clustered as any one of normal/healthy or weak based on his/her input values. (100 rows)

# Aim:
To implement the K-means clustering algorithm to classify patients into three distinct clusters based on their health status. The classification categories are "Normal," "Healthy," and "Weak." By using the K-means clustering method, we aim to group patients with similar characteristics into these categories to aid in understanding patient health distributions and support targeted health interventions.
# Sample Dataset:
BMI	Exercise_Frequency_per_Week	Cluster	Cluster_Label
22.993428306022466	3	2	Healthy	
21.72347139765763	4	2	Healthy	
23.295377076201383	3	2	Healthy	
25.04605971281605	3	2	Healthy	
21.531693250553328	3	2	Healthy	

# Code:
import pandas as pd
from sklearn.cluster import KMeans

data = pd.read_csv(r"C:\Users\adity\Documents\datamining\patient_clustering_data.csv")
X = data[['BMI', 'Exercise_Frequency_per_Week']]

kmeans = KMeans(n_clusters=3, random_state=42)
data['Cluster'] = kmeans.fit_predict(X)

centroids = kmeans.cluster_centers_

data['Cluster_Label'] = data['Cluster'].map({0: 'Weak', 1: 'Normal', 2: 'Healthy'})

data[['BMI', 'Exercise_Frequency_per_Week', 'Cluster_Label']].head(20)

# Output:
 ![image](https://github.com/user-attachments/assets/6927eb2b-0886-4aa7-b5c8-6319e863ce5a)

# Result:
The K-means clustering algorithm successfully categorized patients into three clusters:
1.	Cluster 1: Normal
2.	Cluster 2: Healthy
3.	Cluster 3: Weak
Each cluster represents distinct patient characteristics, providing clear groupings based on health status. The clustering effectively groups patients into these categories, aiding in health data analysis.
# Q3:- The ID3 algorithm builds decision trees using a top-down greedy search approach through the space of possible branches with no backtracking. Consider a dataset of 50 rows “Road transport records” with the attributes “Road ID”, “Length”, Numberof_Bends”, “Trafficvolume” and “AccidentRisk”. Implement the same to the dataset to recommend the decision tree to classify the data.
# Aim:
To implement the ID3 algorithm to build a decision tree for classifying road transport records based on attributes such as "Road ID," "Length," "Numberof_Bends," "Trafficvolume," and "AccidentRisk." The goal is to create a decision tree that can effectively predict the "AccidentRisk" based on the given attributes, aiding in the assessment and management of road safety.
# Sample Dataset:

Road_ID	Length	Numberof_Bends	Trafficvolume	AccidentRisk
R1	7	5	783	Medium	
R2	15	5	971	Low	
R3	11	9	825	Low	
R4	8	3	646	Medium	
R5	7	5	838	Medium	

# Code:
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import tree
import matplotlib.pyplot as plt

df = pd.read_csv(r"C:\Users\adity\Documents\datamining\road_transport_records.csv")

X = df[['Length', 'Numberof_Bends', 'Trafficvolume']]  # Features
y = df['AccidentRisk']  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = DecisionTreeClassifier(criterion='entropy', random_state=42)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

accuracy = clf.score(X_test, y_test)
print(f"Accuracy: {accuracy}")

plt.figure(figsize=(12,8))
tree.plot_tree(clf, feature_names=['Length', 'Numberof_Bends', 'Trafficvolume'], class_names=True, filled=True)
plt.show()
Output:
 ![image](https://github.com/user-attachments/assets/792dd4ed-e144-4a1a-9161-3f9b851e6489)

# Result:
The ID3 algorithm built a decision tree to classify "AccidentRisk" based on attributes like "Length," "Numberof_Bends," and "Trafficvolume." The tree effectively predicts risk levels, providing a clear model for road safety classification.
