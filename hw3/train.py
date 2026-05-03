from typing import cast
import argparse

import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
import csv
import matplotlib.pyplot as plt

from lstm import GRU, LSTM, StackedLSTM

winddir_idx = {
    'SE': 0.0,
    'NW': 1.0,
    'cv': 2.0,
    'NE': 3.0,
}


class TorchLSTM(nn.Module):

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=1,
                            batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class StandardizedTensorDataset(torch.utils.data.TensorDataset):

    def __init__(self, x_tensor: torch.Tensor, y_tensor: torch.Tensor,
                 targets_mean: float, targets_std: float):
        super().__init__(x_tensor, y_tensor)
        self.targets_mean = targets_mean
        self.targets_std = targets_std


def train(model: nn.Module, train_loader: DataLoader, num_epochs: int,
          learning_rate: float, device: torch.device) -> list[float]:
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_history = []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            update_lr(optimizer, learning_rate * (0.99**epoch))

        avg_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')
        loss_history.append(avg_loss)

    return loss_history


def test(model: nn.Module,
         test_loader: DataLoader,
         device: torch.device,
         *,
         run_tag: str = "") -> float:
    model.to(device)
    model.eval()
    predictions = []
    targets_list = []

    dataset = cast(StandardizedTensorDataset, test_loader.dataset)
    targets_mean = dataset.targets_mean
    targets_std = dataset.targets_std

    with torch.no_grad():
        for inputs, target in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            predictions.extend(outputs.cpu().numpy())
            targets_list.extend(target.cpu().numpy())

    predictions = torch.tensor(predictions) * targets_std + targets_mean
    targets_list = torch.tensor(targets_list) * targets_std + targets_mean

    rmse = torch.sqrt(nn.MSELoss()(predictions, targets_list))
    print(f'RMSE: {rmse.item():.4f}')

    model_name = model.__class__.__name__
    y_true = targets_list.detach().cpu().numpy().ravel()
    y_pred = predictions.detach().cpu().numpy().ravel()

    plt.figure(figsize=(18, 5.5))
    plt.plot(y_true, c='r', alpha=0.90, linewidth=2.0)
    plt.plot(y_pred, c='darkblue', alpha=0.75, linewidth=2.0)
    plt.ylabel('Value')
    plt.xlabel('Time Step')
    plt.title(f'{model_name} Time Series: True vs Predicted')
    plt.legend(['True Values', 'Predicted Values'])
    plt.tight_layout()
    suffix = f'_{run_tag}' if run_tag else ''
    plt.savefig(f'{model_name}_true_vs_pred_curve{suffix}.png', dpi=400)
    plt.close()

    return float(rmse.item())


def parse_field(s: str) -> float:
    s = s.strip()
    try:
        return float(s)
    except ValueError:
        if s in winddir_idx:
            return float(winddir_idx[s])
        raise ValueError(f"Unknown string: {s}")


def load_data(
    data_dir: str,
    window_size: int,
    batch_size: int,
    is_test: bool = False,
    stats: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    | None = None,
    shuffle: bool = False,
) -> tuple[DataLoader, tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                             torch.Tensor]]:
    data = []
    with open(data_dir, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            if is_test:
                pollution = row[-1]
                others = row[:-1]
                features = [parse_field(pollution)
                            ] + [parse_field(i) for i in others]
                target = float(row[-1])
            else:
                features = [parse_field(i) for i in row[1:]]
                target = float(row[1])
            data.append((features, target))

    inputs = torch.tensor([item[0] for item in data], dtype=torch.float32)
    targets = torch.tensor([item[1] for item in data],
                           dtype=torch.float32).unsqueeze(1)

    if stats is None:
        inputs_mean = inputs.mean(dim=0, keepdim=True)
        inputs_std = inputs.std(dim=0, keepdim=True)
        inputs_std[inputs_std == 0] = 1.0

        targets_mean = targets.mean(dim=0, keepdim=True)
        targets_std = targets.std(dim=0, keepdim=True)
        targets_std[targets_std == 0] = 1.0
        stats = (inputs_mean, inputs_std, targets_mean, targets_std)
    else:
        inputs_mean, inputs_std, targets_mean, targets_std = stats

    inputs = (inputs - inputs_mean) / inputs_std
    targets = (targets - targets_mean) / targets_std

    x, y = [], []
    for i in range(len(inputs) - window_size):
        x.append(inputs[i:i + window_size])
        y.append(targets[i + window_size])

    x_tensor = torch.stack(x)  # b, window, data_size
    y_tensor = torch.stack(y)  # b, 1

    dataset = StandardizedTensorDataset(x_tensor,
                                        y_tensor,
                                        targets_mean=targets_mean.item(),
                                        targets_std=targets_std.item())
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle), stats


def update_lr(optimizer: optim.Optimizer, new_lr: float) -> None:
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Sweep different window sizes for time-series forecasting')
    parser.add_argument('--window-sizes',
                        type=int,
                        nargs='+',
                        default=[1, 3, 6, 12, 24, 36],
                        help='List of window sizes to evaluate')
    parser.add_argument('--num-epochs', type=int, default=100)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--hidden-size', type=int, default=128)
    parser.add_argument('--models',
                        type=str,
                        nargs='+',
                        default=['lstm', 'stacked_lstm', 'gru'],
                        choices=['lstm', 'stacked_lstm', 'gru'],
                        help='Custom models to compare')
    parser.add_argument('--num-layers',
                        type=int,
                        default=2,
                        help='Number of layers for stacked_lstm')
    parser.add_argument('--train-path',
                        type=str,
                        default='hw3/LSTM-Multivariate_pollution.csv')
    parser.add_argument('--test-path',
                        type=str,
                        default='hw3/pollution_test_data1.csv')
    return parser.parse_args()


def build_model(model_name: str, *, input_size: int, hidden_size: int,
                output_size: int, num_layers: int) -> nn.Module:
    if model_name == 'lstm':
        return LSTM(input_size=input_size,
                    hidden_size=hidden_size,
                    output_size=output_size)
    if model_name == 'stacked_lstm':
        return StackedLSTM(input_size=input_size,
                           hidden_size=hidden_size,
                           output_size=output_size,
                           num_layers=num_layers)
    if model_name == 'gru':
        return GRU(input_size=input_size,
                   hidden_size=hidden_size,
                   output_size=output_size)
    raise ValueError(f'Unknown model: {model_name}')


if __name__ == "__main__":
    args = parse_args()

    input_size = 8
    hidden_size = args.hidden_size
    output_size = 1
    num_epochs = args.num_epochs
    learning_rate = args.learning_rate
    batch_size = args.batch_size

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    results: list[tuple[int, str, float]] = []
    loss_histories: dict[tuple[int, str], list[float]] = {}
    for window_size in args.window_sizes:
        print(f"\n========== window_size = {window_size} ==========")

        train_loader, stats = load_data(args.train_path,
                                        window_size=window_size,
                                        batch_size=batch_size,
                                        is_test=False,
                                        stats=None,
                                        shuffle=True)
        test_loader, _ = load_data(args.test_path,
                                   window_size=window_size,
                                   batch_size=batch_size,
                                   is_test=True,
                                   stats=stats,
                                   shuffle=True)

        for model_name in args.models:
            print(f"\n----- model = {model_name} -----")
            model = build_model(model_name,
                                input_size=input_size,
                                hidden_size=hidden_size,
                                output_size=output_size,
                                num_layers=args.num_layers)
            loss_history = train(model, train_loader, num_epochs,
                                 learning_rate, device)
            loss_histories[(window_size, model_name)] = loss_history
            rmse = test(model,
                        test_loader,
                        device,
                        run_tag=f'{model_name}_win{window_size}')
            results.append((window_size, model_name, rmse))

    if loss_histories:
        plt.figure(figsize=(10.5, 6.0))
        for (window_size, model_name) in sorted(loss_histories.keys()):
            plt.plot(loss_histories[(window_size, model_name)],
                     linewidth=2.0,
                     alpha=0.9,
                     label=f'{model_name}, win={window_size}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Comparison (models & window sizes)')
        plt.legend()
        plt.tight_layout()
        plt.savefig('training_loss_all_windows.png', dpi=400)
        plt.close()

    results_sorted = sorted(results, key=lambda x: (x[0], x[1]))
    print("\n========== Summary (window_size, model -> RMSE) ==========")
    for window_size, model_name, rmse in results_sorted:
        print(f"{window_size:>4}, {model_name:<12} -> {rmse:.4f}")
