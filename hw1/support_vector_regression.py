import os

import torch
import pandas as pd
import matplotlib.pyplot as plt

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def evaluate(truth: torch.Tensor, pred: torch.Tensor) -> tuple[float, float]:
    num = torch.sum((truth - pred)**2)
    den = torch.sum((truth - torch.mean(truth))**2)
    r2 = 1 - (num / den).item()

    rmse = torch.sqrt(num / len(truth)).item()
    return r2, rmse


def support_vector_regression(train: tuple[torch.Tensor, torch.Tensor],
                              test: tuple[torch.Tensor, torch.Tensor]):
    x_train, y_train = train
    x_test, y_test = test

    from sklearn.svm import SVR
    svr = SVR(kernel='rbf', C=1e4, gamma=0.5)
    svr.fit(x_train.reshape(-1, 1).numpy(), y_train.numpy())
    
    predict_train = torch.tensor(svr.predict(x_train.reshape(-1, 1).numpy()))
    predict_test = torch.tensor(svr.predict(x_test.reshape(-1, 1).numpy()))
    return evaluate(y_train, predict_train), evaluate(y_test, predict_test), predict_train.tolist(), predict_test.tolist()


if __name__ == "__main__":
    train = pd.read_excel(r'hw1/hw1_data.xlsx', sheet_name='train')
    test = pd.read_excel(r'hw1/hw1_data.xlsx', sheet_name='test')

    x_train = torch.tensor(train['x'].values)  # b
    y_train = torch.tensor(train['y'].values)
    x_test = torch.tensor(test['x'].values)
    y_test = torch.tensor(test['y'].values)

    (r2_train, rmse_train), (
        r2_test,
        rmse_test), train_predict, test_predict = support_vector_regression(
            (x_train, y_train), (x_test, y_test))
    # print(f"Support Vector Regression (n={n}):")
    print(
        f"Support Vector Regression (Train): R2 = {r2_train:.4f}, RMSE = {rmse_train:.4f}"
    )
    print(
        f"Support Vector Regression (Test): R2 = {r2_test:.4f}, RMSE = {rmse_test:.4f}"
    )

    plt.figure(figsize=(20, 6))
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 20

    plt.subplot(1, 2, 1)
    plt.scatter(train['x'], train['y'], color='blue', label='train')
    plt.plot(train['x'],
             train_predict,
             color='orange',
             label='support vector regression')
    plt.axis('scaled')
    plt.ylim(-3, 3)
    plt.title('Train')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc='upper left')

    plt.subplot(1, 2, 2)
    plt.scatter(test['x'], test['y'], color='red', label='test')
    plt.plot(test['x'],
             test_predict,
             color='orange',
             label='support vector regression')
    plt.axis('scaled')
    plt.ylim(-3, 3)
    plt.title('Test')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc='upper left')
    plt.tight_layout()

    plt.savefig('hw1/tex/figure/support_vector_regression.png', dpi=400)
    plt.show()
