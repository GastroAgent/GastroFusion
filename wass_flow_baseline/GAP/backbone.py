import torch.nn as nn
from torch.func import functional_call

class ConvolutionalNeuralNetwork(nn.Module):
    def __init__(self, in_channels, out_features, hidden_size=64):
        super().__init__()
        def block(a,b):
            return nn.Sequential(
                nn.Conv2d(a,b,3,padding=1),
                nn.BatchNorm2d(b, momentum=1., track_running_stats=False),
                nn.ReLU(),
                nn.MaxPool2d(2)
            )
        self.features=nn.Sequential(
            block(in_channels, hidden_size),
            block(hidden_size, hidden_size),
            block(hidden_size, hidden_size),
            block(hidden_size, hidden_size)
        )
        self.feature_dim = 1600*(hidden_size//64)
        self.classifier=nn.Linear(self.feature_dim, out_features)

    def reset_classifier(self, num_classes):
        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def forward_features(self, x):
        f=self.features(x)
        return f.view(f.size(0), -1)

    def forward(self,x,params=None):
        if params is None:
            return self.classifier(self.forward_features(x))
        return functional_call(self, params, (x,))
