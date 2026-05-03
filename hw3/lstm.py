from typing import Any

import torch
from torch import nn


def make_linear(in_features, out_features, gain=1.41421):
    layer = nn.Linear(in_features, out_features)
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.zeros_(layer.bias)
    return layer


class LSTMCell(nn.Module):

    def __init__(self, *args: Any, input_size: int, hidden_size: int,
                 **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        hidden_state_size = hidden_size + input_size

        # forget gate
        self.wf = make_linear(in_features=hidden_state_size,
                              out_features=hidden_size)

        # input gate
        self.wi_i = make_linear(in_features=hidden_state_size,
                                out_features=hidden_size)
        self.wi_c = make_linear(in_features=hidden_state_size,
                                out_features=hidden_size)

        # output gate
        self.wo = make_linear(in_features=hidden_state_size,
                              out_features=hidden_size)

    def forward(self, x: torch.Tensor, hidden_state: torch.Tensor,
                memory: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        contact_hidden = torch.cat((hidden_state, x), dim=-1)
        memory = memory * torch.sigmoid(self.wf(contact_hidden))
        memory = memory + torch.sigmoid(
            self.wi_i(contact_hidden)) * torch.tanh(self.wi_c(contact_hidden))
        hidden_state = torch.sigmoid(
            self.wo(contact_hidden)) * torch.tanh(memory)
        return hidden_state, memory


class LSTM(nn.Module):

    def __init__(self, *args: Any, input_size: int, hidden_size: int,
                 output_size: int, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self.lstm_cell = LSTMCell(input_size=input_size,
                                  hidden_size=hidden_size)
        self.output_layer = nn.Linear(in_features=hidden_size,
                                      out_features=output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (b, seq_len, data_size)
        seq_len = x.shape[1]
        hidden_state = torch.zeros(
            (x.shape[0], self.lstm_cell.wf.out_features), device=x.device)
        memory = torch.zeros_like(hidden_state)

        for idx in range(seq_len):
            hidden_state, memory = self.lstm_cell(x[:, idx, :], hidden_state,
                                                  memory)

        output = self.output_layer(hidden_state)
        return output


class StackedLSTM(nn.Module):

    def __init__(self,
                 *args: Any,
                 input_size: int,
                 hidden_size: int,
                 output_size: int,
                 num_layers: int = 2,
                 **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        if num_layers < 1:
            raise ValueError('num_layers must be >= 1')

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        cells: list[LSTMCell] = []
        for layer_idx in range(num_layers):
            layer_input_size = input_size if layer_idx == 0 else hidden_size
            cells.append(
                LSTMCell(input_size=layer_input_size, hidden_size=hidden_size))
        self.cells = nn.ModuleList(cells)

        self.output_layer = nn.Linear(in_features=hidden_size,
                                      out_features=output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (b, seq_len, data_size)
        batch_size, seq_len, _ = x.shape

        hidden_states = [
            torch.zeros((batch_size, self.hidden_size), device=x.device)
            for _ in range(self.num_layers)
        ]
        memories = [
            torch.zeros_like(hidden_states[0]) for _ in range(self.num_layers)
        ]

        for t in range(seq_len):
            layer_input = x[:, t, :]
            for layer_idx, cell in enumerate(self.cells):
                h, c = cell(layer_input, hidden_states[layer_idx],
                            memories[layer_idx])
                hidden_states[layer_idx] = h
                memories[layer_idx] = c
                layer_input = h

        return self.output_layer(hidden_states[-1])


class GRUCell(nn.Module):

    def __init__(self, *args: Any, input_size: int, hidden_size: int,
                 **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        hidden_state_size = hidden_size + input_size

        self.wz = make_linear(in_features=hidden_state_size,
                              out_features=hidden_size)
        self.wr = make_linear(in_features=hidden_state_size,
                              out_features=hidden_size)

        # candidate hidden uses [r*h, x]
        self.wh = make_linear(in_features=hidden_state_size,
                              out_features=hidden_size)

        self.hidden_size = hidden_size

    def forward(self, x: torch.Tensor,
                hidden_state: torch.Tensor) -> torch.Tensor:
        contact_hidden = torch.cat((hidden_state, x), dim=-1)
        z = torch.sigmoid(self.wz(contact_hidden))
        r = torch.sigmoid(self.wr(contact_hidden))
        candidate_in = torch.cat((r * hidden_state, x), dim=-1)
        h_tilde = torch.tanh(self.wh(candidate_in))
        hidden_state = (1.0 - z) * h_tilde + z * hidden_state
        return hidden_state


class GRU(nn.Module):

    def __init__(self, *args: Any, input_size: int, hidden_size: int,
                 output_size: int, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self.gru_cell = GRUCell(input_size=input_size, hidden_size=hidden_size)
        self.output_layer = nn.Linear(in_features=hidden_size,
                                      out_features=output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (b, seq_len, data_size)
        batch_size, seq_len, _ = x.shape
        hidden_state = torch.zeros((batch_size, self.gru_cell.hidden_size),
                                   device=x.device)

        for t in range(seq_len):
            hidden_state = self.gru_cell(x[:, t, :], hidden_state)

        return self.output_layer(hidden_state)
