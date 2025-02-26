import pickle
import numpy as np
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data_dict=pickle.load(open('./data.pickle','rb'))
data=np.asarray(data_dict['data'])
labels=np.asarray(data_dict['labels'])

class_counts = Counter(labels)
print(class_counts)

x_train,x_test,y_train,y_test=train_test_split(data,labels,test_size=0.2,shuffle=True,stratify=labels)

model=RandomForestClassifier()
model.fit(x_train,y_train)
y_predict=model.predict(x_test)

score=accuracy_score(y_predict,y_test)
print(f'{score} Is Acurracy');

f=open('model.p','wb');
pickle.dump({'model':model},f)
f.close()
