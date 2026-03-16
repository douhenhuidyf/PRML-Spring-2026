import pandas as pd
import torch
import matplotlib.pyplot as plt


def evaluate(truth: torch.Tensor, pred: torch.Tensor) -> tuple[float, float]:
    num = torch.sum((truth - pred)**2)
    den = torch.sum((truth - torch.mean(truth))**2)
    r2 = 1 - (num / den).item()

    rmse = torch.sqrt(num / len(truth)).item()
    return r2, rmse


def least_squares(x: torch.Tensor,
                  y: torch.Tensor) -> tuple[tuple[float, float], list[float]]:
    x = x.reshape(-1, 1)  # b, 1
    x = torch.hstack((torch.ones((x.shape[0], 1)), x))  # b, 2
    beta = torch.linalg.inv(x.T @ x) @ x.T @ y  # 2,
    return evaluate(y, x @ beta), beta.tolist()


def gradient_descent(
        x: torch.Tensor,
        y: torch.Tensor,
        lr: float = 2e-4,
        iter_num: int = 100000) -> tuple[tuple[float, float], list[float]]:
    x = x.reshape(-1, 1)  # b, 1
    x = torch.hstack((torch.ones((x.shape[0], 1)), x))  # b, 2
    beta = torch.ones((x.shape[1], ), dtype=x.dtype)  # 2,
    for _ in range(iter_num):
        pred = x @ beta
        grad = -2 * x.T @ (y - pred) / len(y)
        beta -= lr * grad
    return evaluate(y, x @ beta), beta.tolist()


def newton_method(
        x: torch.Tensor,
        y: torch.Tensor,
        iter_num: int = 100) -> tuple[tuple[float, float], list[float]]:
    x = x.reshape(-1, 1)  # b, 1
    x = torch.hstack((torch.ones((x.shape[0], 1)), x))  # b, 2
    beta = torch.ones((x.shape[1], ), dtype=x.dtype)  # 2,
    for _ in range(iter_num):
        pred = x @ beta
        grad = -2 * x.T @ (y - pred) / len(y)
        hessian = 2 * x.T @ x / len(y)
        beta -= torch.linalg.solve(hessian, grad)
    return evaluate(y, x @ beta), beta.tolist()


if __name__ == "__main__":
    train = pd.read_excel(r'hw1/hw1_data.xlsx', sheet_name='train')
    test = pd.read_excel(r'hw1/hw1_data.xlsx', sheet_name='test')

    X_train = torch.tensor(train['x'].values)  # b
    y_train = torch.tensor(train['y'].values)
    X_test = torch.tensor(test['x'].values)
    y_test = torch.tensor(test['y'].values)

    for func in [least_squares, gradient_descent, newton_method]:
        (r2_train, rmse_train), beta_train = func(X_train, y_train)
        (r2_test, rmse_test), beta_test = func(X_test, y_test)
        print(f"{func.__name__}:")
        print(f"  Train R^2: {r2_train}, RMSE: {rmse_train}")
        print(f"  Test R^2: {r2_test}, RMSE: {rmse_test}")

    # draw the line of linear regression fitted by least squares method on train and test data
    plt.figure(figsize=(10, 6))
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 12

    plt.subplot(1, 2, 1)
    plt.scatter(train['x'], train['y'], color='blue', label='train')
    plt.plot(train['x'],
             beta_train[0] + beta_train[1] * train['x'],
             color='orange',
             label='linear regression')
    plt.title('Train Data')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.scatter(test['x'], test['y'], color='red', label='test')
    plt.plot(test['x'],
             beta_test[0] + beta_test[1] * test['x'],
             color='orange',
             label='linear regression')
    plt.title('Test Data')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.tight_layout()

    plt.savefig('hw1/tex/figure/linear_regression.png', dpi=400)
    plt.show()
