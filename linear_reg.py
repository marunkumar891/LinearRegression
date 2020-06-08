import pandas as pd
import numpy as np
from sklearn import model_selection as ms
from sklearn import linear_model as lm
from matplotlib import pyplot as plt

data = pd.read_csv("student-mat.csv",sep=';')

print(data.head())
print(data.describe())
print(data.columns)

print(data['G1'].value_counts())
plt.scatter(data['G1'],data['G3'])
plt.xlabel('Grade-1')
plt.ylabel('Grade-3')
plt.title('Grade-1 vs Grade-3')
plt.show()

print(data['G2'].value_counts())
plt.scatter(data['G2'],data['G3'])
plt.xlabel('Grade-2')
plt.ylabel('Grade-3')
plt.title('Grade-2 vs Grade-3')
plt.show()


print(data['studytime'].value_counts())
plt.bar(data['studytime'],data['G3'])
plt.xlabel('studytime')
plt.ylabel('Grade-3')
plt.title('studytime vs Grade-3')
plt.show()


print(data['failures'].value_counts())
plt.bar(data['failures'],data['G3'])
plt.xlabel('failures')
plt.ylabel('Grade-3')
plt.title('failures vs Grade-3')
plt.show()


print(data['absences'].value_counts())
plt.scatter(data['absences'],data['G3'])
plt.xlabel('absences')
plt.ylabel('Grade-3')
plt.title('absences vs Grade-3')
plt.show()


data = data[['G1','G2','G3','studytime','failures','absences']]

predict = 'G3'

x = np.array(data.drop([predict],1))
y = np.array(data[predict])

x_train, x_test, y_train, y_test = ms.train_test_split(x,y,test_size=0.1)

linear = lm.LinearRegression()

linear.fit(x_train,y_train)

acc = linear.score(x_test, y_test)

print('\n')
print('accuracy of prediction:',round(acc*100,2) ,'%\n')

prediction = linear.predict(x_test)

print('Original value vs Predicted Value')
for i in range(len(prediction)):
    print(round(prediction[i],3),'\t', y_test[i])


