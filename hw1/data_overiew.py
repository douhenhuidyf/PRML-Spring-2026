import matplotlib.pyplot as plt
import pandas as pd

# draw a pic of train and test x, y dotplot from two table of .xlsx to see the distribution of data

train = pd.read_excel(r'hw1/hw1_data.xlsx', sheet_name='train')
test = pd.read_excel(r'hw1/hw1_data.xlsx', sheet_name='test')

plt.figure(figsize=(10, 6))
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12

plt.subplot(1, 2, 1)
plt.scatter(train['x'], train['y'], color='blue', label='train')
plt.title('Train Data')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.subplot(1, 2, 2)
plt.scatter(test['x'], test['y'], color='red', label='test')
plt.title('Test Data')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.tight_layout()

plt.savefig('hw1/tex/figure/data_distribution.png', dpi=400)
plt.show()
