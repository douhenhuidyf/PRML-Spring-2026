import os

import torch
import pandas as pd
import matplotlib.pyplot as plt

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def evaluate(truth: torch.Tensor,
             predict: torch.Tensor) -> tuple[float, float]:
    num = torch.sum((truth - predict)**2)
    den = torch.sum((truth - torch.mean(truth))**2)
    r2 = 1 - (num / den).item()

    rmse = torch.sqrt(num / len(truth)).item()
    return r2, rmse


def kernel_regression(
    train: tuple[torch.Tensor,
                 torch.Tensor], test: tuple[torch.Tensor,
                                            torch.Tensor], bandwidth: float
) -> tuple[tuple[float, float], tuple[float, float], list[float], list[float]]:
    x_train, y_train = train
    x_test, y_test = test

    weight = torch.exp(-0.5 *
                       (x_train.reshape(-1, 1) - x_train.reshape(1, -1))**2 /
                       (bandwidth**2))
    weight = weight / torch.sum(weight, dim=1, keepdim=True)
    train_predict = weight @ y_train
    weight = torch.exp(-0.5 *
                       (x_train.reshape(-1, 1) - x_test.reshape(1, -1))**2 /
                       (bandwidth**2))
    weight = weight / torch.sum(weight, dim=1, keepdim=True)
    test_predict = weight @ y_train

    train_eval = evaluate(y_train, train_predict)
    test_eval = evaluate(y_test, test_predict)
    return train_eval, test_eval, train_predict.tolist(), test_predict.tolist()


if __name__ == "__main__":
    train = pd.read_excel(r'hw1/hw1_data.xlsx', sheet_name='train')
    test = pd.read_excel(r'hw1/hw1_data.xlsx', sheet_name='test')

    x_train = torch.tensor(train['x'].values)  # b
    y_train = torch.tensor(train['y'].values)
    x_test = torch.tensor(test['x'].values)
    y_test = torch.tensor(test['y'].values)

    for bandwidth in [0.1, 0.25, 0.5, 1.0, 1.5]:
        (r2_train, rmse_train), (
            r2_test,
            rmse_test), train_predict, test_predict = kernel_regression(
                (x_train, y_train), (x_test, y_test), bandwidth)
        print(f"Kernel Regression (bandwidth={bandwidth}):")
        print(
            f"Kernel Regression (Train): R2 = {r2_train:.4f}, RMSE = {rmse_train:.4f}"
        )
        print(
            f"Kernel Regression (Test): R2 = {r2_test:.4f}, RMSE = {rmse_test:.4f}"
        )

        plt.figure(figsize=(20, 6))
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['font.size'] = 20

        plt.subplot(1, 2, 1)
        plt.scatter(train['x'], train['y'], color='blue', label='train')
        plt.plot(train['x'],
                 train_predict,
                 color='orange',
                 label='kernel regression')
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
                 label='kernel regression')
        plt.axis('scaled')
        plt.ylim(-3, 3)
        plt.title('Test')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend(loc='upper left')
        plt.tight_layout()

        plt.savefig('hw1/tex/figure/kernel_regression_' + str(bandwidth) +
                    '.png',
                    dpi=400)
        # plt.show()
