from torch import nn, optim

def get_model():
    class MyModel(nn.Module):
        def __init__(self, n_input, n_output):
            super().__init__()

            self.model = nn.Sequential(
                nn.Linear(n_input, 256),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(256, n_output),
            )

        def forward(self, x):
            x = self.model(x)
            return x
        
    return MyModel(n_input=784, n_output=10)