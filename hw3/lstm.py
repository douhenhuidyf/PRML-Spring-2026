from typing import Any

import torch
from torch import nn


class LSTMCell(nn.Module):

    def __init__(self, *args: Any, input_size: int, hidden_size: int,
                 output_size: int, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        hidden_state_size = hidden_size + input_size

        # forget gate
        self.wf = nn.Linear(in_features=hidden_state_size,
                            out_features=hidden_size)

        # input gate
        self.wi_i = nn.Linear(in_features=hidden_state_size,
                              out_features=hidden_size)
        self.wi_c = nn.Linear(in_features=hidden_state_size,
                              out_features=hidden_size)

        # output gate
        self.wo = nn.Linear(in_features=hidden_state_size,
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
                                  hidden_size=hidden_size,
                                  output_size=output_size)
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
