from torch import nn
import torch


class MaskModel(nn.Module):
    def __init__(self, input_shape=(32, 325, 96), threshold=0.5, patch=4,
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        super().__init__()
        self.threshold = torch.tensor(threshold, device=device)

        # a learnable mask matrix
        self.mask = nn.Parameter(torch.empty((input_shape[0], input_shape[1], input_shape[2]), device=device))
        torch.nn.init.xavier_normal_(self.mask)

        self.indices_to_keep = (torch.arange(input_shape[1], device=device) % patch == 0).unsqueeze(-1).float()

    def forward(self, x):
        mask = torch.sigmoid(self.mask)
        mask = mask * (1 - self.indices_to_keep) + self.indices_to_keep
        mask = torch.sigmoid((mask - self.threshold) * 64)
        z = x * mask
        return z
