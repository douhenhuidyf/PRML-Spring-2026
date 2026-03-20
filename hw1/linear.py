import os

import pandas as pd
import torch
import matplotlib.pyplot as plt

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def evaluate(truth: torch.Tensor, pred: torch.Tensor) -> tuple[float, float]:
    num = torch.sum((truth - pred)**2)
    den = torch.sum((truth - torch.mean(truth))**2)
    r2 = 1 - (num / den).item()

    rmse = torch.sqrt(num / len(truth)).item()
    return r2, rmse


def least_squares(
    train: tuple[torch.Tensor, torch.Tensor], test: tuple[torch.Tensor,
                                                          torch.Tensor]
) -> tuple[tuple[float, float], tuple[float, float], list[float], list[float]]:
    train_x, train_y = train
    test_x, test_y = test

    train_x = train_x.reshape(-1, 1)  # b, 1
    train_x = torch.hstack((torch.ones(
        (train_x.shape[0], 1)), train_x))  # b, 2
    test_x = test_x.reshape(-1, 1)  # b, 1
    test_x = torch.hstack((torch.ones((test_x.shape[0], 1)), test_x))  # b, 2
    beta = torch.linalg.inv(train_x.T @ train_x) @ train_x.T @ train_y  # 2,

    predict_train = train_x @ beta
    predict_test = test_x @ beta
    train_eval = evaluate(train_y, predict_train)
    test_eval = evaluate(test_y, predict_test)
    return train_eval, test_eval, predict_train.tolist(), predict_test.tolist()


def gradient_descent(
    train: tuple[torch.Tensor, torch.Tensor],
    test: tuple[torch.Tensor, torch.Tensor],
    lr: float = 1e-3,
    iter_num: int = 10000
) -> tuple[tuple[float, float], tuple[float, float], list[float], list[float]]:
    train_x, train_y = train
    test_x, test_y = test

    train_x = train_x.reshape(-1, 1)  # b, 1
    train_x = torch.hstack((torch.ones(
        (train_x.shape[0], 1)), train_x))  # b, 2
    test_x = test_x.reshape(-1, 1)  # b, 1
    test_x = torch.hstack((torch.ones((test_x.shape[0], 1)), test_x))  # b, 2
    beta = torch.ones((train_x.shape[1], ), dtype=train_x.dtype)  # 2,
    for _ in range(iter_num):
        pred = train_x @ beta
        grad = -2 * train_x.T @ (train_y - pred) / len(train_y)
        beta -= lr * grad

    predict_train = train_x @ beta
    predict_test = test_x @ beta
    train_eval = evaluate(train_y, predict_train)
    test_eval = evaluate(test_y, predict_test)
    return train_eval, test_eval, predict_train.tolist(), predict_test.tolist()


def newton_method(
    train: tuple[torch.Tensor, torch.Tensor],
    test: tuple[torch.Tensor, torch.Tensor],
    iter_num: int = 100
) -> tuple[tuple[float, float], tuple[float, float], list[float], list[float]]:
    train_x, train_y = train
    test_x, test_y = test

    train_x = train_x.reshape(-1, 1)  # b, 1
    train_x = torch.hstack((torch.ones(
        (train_x.shape[0], 1)), train_x))  # b, 2
    test_x = test_x.reshape(-1, 1)  # b, 1
    test_x = torch.hstack((torch.ones((test_x.shape[0], 1)), test_x))  # b, 2
    beta = torch.ones((train_x.shape[1], ), dtype=train_x.dtype)  # 2,
    for _ in range(iter_num):
        pred = train_x @ beta
        grad = -2 * train_x.T @ (train_y - pred) / len(train_y)
        hessian = 2 * train_x.T @ train_x / len(train_y)
        beta -= torch.linalg.solve(hessian, grad)

    predict_train = train_x @ beta
    predict_test = test_x @ beta
    train_eval = evaluate(train_y, predict_train)
    test_eval = evaluate(test_y, predict_test)
    return train_eval, test_eval, predict_train.tolist(), predict_test.tolist()


if __name__ == "__main__":
    train = pd.read_excel(r'hw1/hw1_data.xlsx', sheet_name='train')
    test = pd.read_excel(r'hw1/hw1_data.xlsx', sheet_name='test')

    x_train = torch.tensor(train['x'].values)  # b
    y_train = torch.tensor(train['y'].values)
    x_test = torch.tensor(test['x'].values)
    y_test = torch.tensor(test['y'].values)

    for func in [least_squares, gradient_descent, newton_method]:
        (r2_train,
         rmse_train), (r2_test, rmse_test), train_predict, test_predict = func(
             (x_train, y_train), (x_test, y_test))
        print(f"{func.__name__}:")
        print(f"  Train R^2: {r2_train:.4f}, RMSE: {rmse_train:.4f}")
        print(f"  Test R^2: {r2_test:.4f}, RMSE: {rmse_test:.4f}")

    plt.figure(figsize=(20, 6))
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 20

    plt.subplot(1, 2, 1)
    plt.scatter(train['x'], train['y'], color='blue', label='train')
    plt.plot(train['x'],
             train_predict,
             color='orange',
             label='linear regression')
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
             label='linear regression')
    plt.axis('scaled')
    plt.ylim(-3, 3)
    plt.title('Test')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc='upper left')
    plt.tight_layout()

    plt.savefig('hw1/tex/figure/linear_regression.png', dpi=400)
    plt.show()
