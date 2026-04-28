import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
import csv
import matplotlib.pyplot as plt

from lstm import LSTM

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


def train(model: nn.Module, train_loader: DataLoader, num_epochs: int,
          learning_rate: float, device: torch.device) -> None:
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss = []

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

        avg_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')
        loss.append(avg_loss)

    plt.plot(loss)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.savefig('training_loss.png', dpi=400)


def test(model: nn.Module, test_loader: DataLoader,
         device: torch.device) -> None:
    model.to(device)
    model.eval()
    predictions = []
    targets = []
    with torch.no_grad():
        for inputs, target in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            predictions.extend(outputs.cpu().numpy())
            targets.extend(target.cpu().numpy())
    # calculate RMSE
    predictions = torch.tensor(predictions)
    targets = torch.tensor(targets)
    rmse = torch.sqrt(nn.MSELoss()(predictions, targets))
    print(f'RMSE: {rmse.item():.4f}')


def parse_field(s: str) -> float:
    s = s.strip()
    try:
        return float(s)
    except ValueError:
        if s in winddir_idx:
            return float(winddir_idx[s])
        raise ValueError(f"Unknown string: {s}")


def load_data(data_dir: str,
              window_size: int,
              batch_size: int,
              is_test: bool = False) -> DataLoader:
    data = []
    with open(data_dir, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            if is_test:
                features = [parse_field(i) for i in row[:-1]]
                target = float(row[-1])
            else:
                features = [parse_field(i) for i in row[2:]]
                target = float(row[1])
            data.append((features, target))

    inputs = torch.tensor([item[0] for item in data], dtype=torch.float32)
    targets = torch.tensor([item[1] for item in data],
                           dtype=torch.float32).unsqueeze(1)

    x, y = [], []
    for i in range(len(inputs) - window_size):
        x.append(inputs[i:i + window_size])
        y.append(targets[i + window_size])

    x_tensor = torch.stack(x)  # b, window, data_size
    y_tensor = torch.stack(y)  # b, 1

    dataset = torch.utils.data.TensorDataset(x_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


if __name__ == "__main__":
    input_size = 7
    hidden_size = 128
    output_size = 1
    num_epochs = 1000
    learning_rate = 1e-3
    batch_size = 32
    window_size = 10

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader = load_data('hw3/LSTM-Multivariate_pollution.csv',
                             window_size=window_size,
                             batch_size=batch_size,
                             is_test=False)
    test_loader = load_data('hw3/pollution_test_data1.csv',
                            window_size=window_size,
                            batch_size=batch_size,
                            is_test=True)

    print("========== Training Custom LSTM ==========")
    custom_model = LSTM(input_size=input_size,
                        hidden_size=hidden_size,
                        output_size=output_size)
    train(custom_model, train_loader, num_epochs, learning_rate, device)
    test(custom_model, test_loader, device)

    print("\n========== Training PyTorch Built-in LSTM ==========")
    torch_model = TorchLSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            output_size=output_size)
    train(torch_model, train_loader, num_epochs, learning_rate, device)
    test(torch_model, test_loader, device)
