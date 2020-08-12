import torch.nn as nn
import torch.nn.functional as F

from config import Instance


class Model(nn.Module):
    def __init__(self, instance: Instance):
        super(Model, self).__init__()
        self.first = nn.Linear(4, 50)
        self.hidden = [nn.Linear(50, 50) for _ in range(instance.n_hidden_layers)]
        self.last = nn.Linear(50, 3)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = F.relu(self.first(x))
        for hidden in self.hidden:
            x = F.relu(hidden(x))
        x = self.last(x)
        y = self.softmax(x)
        return y
