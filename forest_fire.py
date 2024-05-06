import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import warnings
import pickle

warnings.filterwarnings("ignore")

data = pd.read_csv("/kaggle/input/ffpredict/Forest_fire.csv")
data = np.array(data)

X = data[1:, 1:-1]
y = data[1:, -1]
y = y.astype('int')
X = X.astype('int')
# print(X,y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
rand_f = RandomForestRegressor()

rand_f.fit(X_train, y_train)

inputt=[int(x) for x in "45 32 60".split(' ')]
final=[np.array(inputt)]

b = rand_f.predict(final)

pickle.dump(rand_f,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))