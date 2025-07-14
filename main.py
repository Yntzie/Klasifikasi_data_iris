import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

#load dataset
dataIris = load_iris()
df = pd.DataFrame(data=dataIris.data, columns=dataIris.feature_names)
df['target'] = dataIris.target
df['target_name'] = df['target'].apply(lambda x:dataIris.target_names[x])

print(df.head)

#visualisasi data
sns.pairplot(df, hue='target_name')
plt.savefig('images/pairplot.png')
plt.show

#split data(8-% training, 20% test)
x = df[dataIris.feature_names]
y = df['target']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#training model
model = DecisionTreeClassifier()
model.fit(x_train,y_train)

#prediksi
y_pred = model.predict(x_test)

print('Akurasi= ', accuracy_score(y_test, y_pred))
print('Klasifikasi: ')
print(classification_report(y_test, y_pred, target_names=dataIris.target_names))