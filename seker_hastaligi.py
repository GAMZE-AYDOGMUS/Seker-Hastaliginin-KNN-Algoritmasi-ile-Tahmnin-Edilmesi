import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Şeker Hastalığının KNN Algoritması ile Tahmin Edilmesi

data=pd.read_csv(r"C:\Users\AYDOGMUS\Downloads\diabetes.csv")
print(data.head())

print()

sekerHastasi=data[data.Outcome==1]
saglikli=data[data.Outcome==0]

print(saglikli)

plt.scatter(saglikli.Age, saglikli.Glucose, color="green", label="sağlıklı", alpha=0.4)
plt.scatter(sekerHastasi.Age, sekerHastasi.Glucose, color="red", label="hasta", alpha=0.4)
plt.xlabel("yaş")
plt.ylabel("glukoz")
plt.legend()
plt.show()

x1=data.iloc[:,0:-1]
y1=data.iloc[:,-1].values

# train ve test verilerine bakalım
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x1,y1,test_size=0.3,random_state=43)

# standartlaştırma işlemi yapalım
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()

xtrain1=sc.fit_transform(xtrain)
xtest1=sc.transform(xtest)

from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(17)     # Komşu sayıları ne kadar fazla olursa, model daha fazla genelleme yapar ama çok yüksek olursa aşırı genelleme olabilir.


knn.fit(xtrain1,ytrain)

yhead=knn.predict(xtest1)
print("KNN modelinin başarısı:")
print(knn.score(xtest1,ytest))

# en iyi k değerinin bulunması
scorelite=[]
for i in range(1,30):
    knn2=KNeighborsClassifier(n_neighbors=i)
    knn2.fit(xtrain1,ytrain)
    scorelite.append(knn2.score(xtest1,ytest))

plt.plot(range(1,30),scorelite)
plt.xlabel("komşu sayısı")
plt.ylabel("doğruluk")
plt.show()

print()

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(ytest,yhead)
print(cm)