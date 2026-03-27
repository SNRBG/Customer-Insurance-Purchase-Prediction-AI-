import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import train_test_split

import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'Social_Network_Ads.csv')

data = pd.read_csv(DATA_PATH)

X = data[['Age', 'EstimatedSalary']]
y = data['Purchased']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0
)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

model = SVC()
model.fit(X_train, y_train)

samples = np.array([
    [30, 87000],
    [40, 0],
    [40, 100000],
    [50, 0]
])

samples_scaled = sc.transform(samples)
predictions = model.predict(samples_scaled)

for s, p in zip(samples, predictions):
    print(f"Age: {s[0]}, Salary: {s[1]} → Purchased: {p}")
