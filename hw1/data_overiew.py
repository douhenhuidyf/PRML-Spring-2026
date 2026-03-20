from turtle import up

import matplotlib.pyplot as plt
import pandas as pd

train = pd.read_excel(r'hw1/hw1_data.xlsx', sheet_name='train')
test = pd.read_excel(r'hw1/hw1_data.xlsx', sheet_name='test')

plt.figure(figsize=(20, 6))
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 20

plt.subplot(1, 2, 1)
plt.scatter(train['x'], train['y'], color='blue', label='train')
plt.axis('scaled')
plt.ylim(-3, 3)
plt.title('Train')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc='upper left')
plt.subplot(1, 2, 2)
plt.scatter(test['x'], test['y'], color='red', label='test')
plt.axis('scaled')
plt.ylim(-3, 3)
plt.title('Test')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc='upper left')
plt.tight_layout()

plt.savefig('hw1/tex/figure/data_distribution.png', dpi=400)
plt.show()
