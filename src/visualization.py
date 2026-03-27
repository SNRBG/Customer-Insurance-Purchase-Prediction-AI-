import pandas as pd
import matplotlib.pyplot as plt

import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'Social_Network_Ads.csv')

data = pd.read_csv(DATA_PATH)

plt.scatter(
    data['Age'],
    data['EstimatedSalary'],
    c=data['Purchased']
)
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.title('Customer Insurance Purchase Behavior')
plt.show()
