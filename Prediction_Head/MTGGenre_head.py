import torch
from torch import nn
import torch.nn.functional as F

class MLPProberBase(nn.Module):
    def __init__(self, d=768, layer='all', num_outputs=87):
        super().__init__()
        
        self.hidden_layer_sizes = [512, ] # eval(self.cfg.hidden_layer_sizes)
        
        self.num_layers = len(self.hidden_layer_sizes)

        self.layer = layer

        for i, ld in enumerate(self.hidden_layer_sizes):
            setattr(self, f"hidden_{i}", nn.Linear(d, ld))
            d = ld
        self.output = nn.Linear(d, num_outputs)

        self.n_tranformer_layer = 12
        
        self.init_aggregator()


    def init_aggregator(self):
        """Initialize the aggregator for weighted sum over different layers of features
        """
        if self.layer == "all":
            # use learned weights to aggregate features
            self.aggregator = nn.Parameter(torch.randn((1, self.n_tranformer_layer, 1)))


    def forward(self, x):
        """
        x: (B, L, T, H)
        T=#chunks, can be 1 or several chunks
        """
        
        if self.layer == "all":
            weights = F.softmax(self.aggregator, dim=1)
            x = (x * weights).sum(dim=1)

        for i in range(self.num_layers):
            x = getattr(self, f"hidden_{i}")(x)
            # x = self.dropout(x)
            x = F.relu(x)
        output = self.output(x)
        return output