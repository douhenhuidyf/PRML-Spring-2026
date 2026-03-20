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


def polynomial_regression(
    train: tuple[torch.Tensor, torch.Tensor], test: tuple[torch.Tensor,
                                                          torch.Tensor], n: int
) -> tuple[tuple[float, float], tuple[float, float], list[float], list[float]]:
    train_x, train_y = train
    test_x, test_y = test

    train_x = train_x.reshape(-1, 1)  # b, 1
    x_poly = torch.hstack([train_x**i for i in range(n + 1)])  # b, n+1
    test_x = test_x.reshape(-1, 1)  # b, 1
    x_poly_test = torch.hstack([test_x**i for i in range(n + 1)])  # b, n+1
    beta = torch.linalg.inv(x_poly.T @ x_poly) @ x_poly.T @ train_y  # n+1,

    train_predict = x_poly @ beta  # b,
    test_predict = x_poly_test @ beta  # b,
    train_eval = evaluate(train_y, train_predict)
    test_eval = evaluate(test_y, test_predict)
    return train_eval, test_eval, train_predict.tolist(), test_predict.tolist()


if __name__ == "__main__":
    train = pd.read_excel(r'hw1/hw1_data.xlsx', sheet_name='train')
    test = pd.read_excel(r'hw1/hw1_data.xlsx', sheet_name='test')

    x_train = torch.tensor(train['x'].values)  # b
    y_train = torch.tensor(train['y'].values)
    x_test = torch.tensor(test['x'].values)
    y_test = torch.tensor(test['y'].values)

    for n in range(1, 16):
        (r2_train, rmse_train), (
            r2_test,
            rmse_test), train_predict, test_predict = polynomial_regression(
                (x_train, y_train), (x_test, y_test), n=n)
        print(f"Polynomial Regression (n={n}):")
        print(
            f"Polynomial Regression (Train): R2 = {r2_train:.4f}, RMSE = {rmse_train:.4f}"
        )
        print(
            f"Polynomial Regression (Test): R2 = {r2_test:.4f}, RMSE = {rmse_test:.4f}"
        )
        if n == 11:
            plt.figure(figsize=(20, 6))
            plt.rcParams['font.family'] = 'Times New Roman'
            plt.rcParams['font.size'] = 20

            plt.subplot(1, 2, 1)
            plt.scatter(train['x'], train['y'], color='blue', label='train')
            plt.plot(train['x'],
                     train_predict,
                     color='orange',
                     label='polynomial regression')
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
                     label='polynomial regression')
            plt.axis('scaled')
            plt.ylim(-3, 3)
            plt.title('Test')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.legend(loc='upper left')
            plt.tight_layout()

            plt.savefig('hw1/tex/figure/polynomial_regression.png', dpi=400)
    plt.show()
