import torch
from torch import nn

class GoogleNet(nn.Module):
    def __init__(self, n_outputs=10, **kwargs):
        """
        The whole architecture of GoogleNet.
        Consist of the previous definition of inception block.
        After each inception block, we can get auxilary loss.
        Give different weight to each loss, make total loss.
        """
        super(GoogleNet, self).__init__(**kwargs)

        self.block1 = nn.Sequential(
            nn.LazyConv2d(out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.LocalResponseNorm(2),
            nn.LazyConv2d(out_channels=64, kernel_size=1),
            nn.ReLU(),
            nn.LazyConv2d(out_channels=192, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.LocalResponseNorm(2),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.block2 = nn.Sequential(
            GoogleNetInception(c1=64, c2=(96, 128), c3=(16, 32), c4=32),
            GoogleNetInception(c1=128, c2=(128, 192), c3=(32, 96), c4=64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            GoogleNetInception(c1=192, c2=(96, 208), c3=(16, 48), c4=64)
        )

        self.block3 = nn.Sequential(
            GoogleNetInception(c1=160, c2=(112, 224), c3=(24, 64), c4=64),
            GoogleNetInception(c1=128, c2=(128, 256), c3=(24, 64), c4=64),
            GoogleNetInception(c1=112, c2=(144, 288), c3=(32, 64), c4=64)
        )

        self.block4 = nn.Sequential(
            GoogleNetInception(c1=256, c2=(160, 320), c3=(32, 128), c4=128),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            GoogleNetInception(c1=256, c2=(160, 320), c3=(32, 128), c4=128),
            GoogleNetInception(c1=384, c2=(192, 384), c3=(48, 128), c4=128),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.LazyLinear(out_features=n_outputs)
        )

        self.aux1 = GoogleNetInceptionAux(n_outputs)
        self.aux2 = GoogleNetInceptionAux(n_outputs)

    def forward(self, x):
        x = self.block1(x)
        out1 = self.block2(x)
        aux1 = self.aux1(out1)
        out2 = self.block3(out1)
        aux2 = self.aux2(out2)
        x = self.block4(out2)
        return x, aux2, aux1


class GoogleNetInception(nn.Module):
    def __init__(self, c1, c2, c3, c4, **kwargs):
        """
        A set of layers.
        This is a part of the whole network.
        Inception consists of 4 ways,
        1st, conv
        2nd, conv, conv
        3rd, conv, conv
        4th, pool, conv.
        """
        super(GoogleNetInception, self).__init__(**kwargs)

        self.b1 = nn.LazyConv2d(out_channels=c1, kernel_size=1)
        
        self.b2_1 = nn.LazyConv2d(out_channels=c2[0], kernel_size=1)
        self.b2_2 = nn.LazyConv2d(out_channels=c2[1], kernel_size=3, padding=1)

        self.b3_1 = nn.LazyConv2d(out_channels=c3[0], kernel_size=1)
        self.b3_2 = nn.LazyConv2d(out_channels=c3[1], kernel_size=3, padding=1)

        self.b4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.b4_2 = nn.LazyConv2d(out_channels=c4, kernel_size=1)

    def forward(self, x):
        """
        The feature map is a concatenation of each result of ways.
        Activation function is ReLU function.
        """
        b1 = torch.relu(self.b1(x))
        b2 = torch.relu(self.b2_2(torch.relu(self.b2_1(x))))
        b3 = torch.relu(self.b3_2(torch.relu(self.b3_1(x))))
        b4 = torch.relu(self.b4_2(self.b4_1(x)))
        return torch.cat((b1, b2, b3, b4), dim=1)


class GoogleNetInceptionAux(nn.Module):
    def __init__(self, n_outputs, **kwargs):
        """
        Using the output of inception block, we will make auxilary scores.
        To get a score vector, we have to connet fully conneted layer at the end.
        Between convolutional layer and fully connected layer, input shape should be flattened.
        """
        super(GoogleNetInceptionAux, self).__init__(**kwargs)

        self.conv = nn.Sequential(
            # nn.AvgPool2d(kernel_size=5, stride=3),
            nn.LazyConv2d(out_channels=128, kernel_size=1)
        )

        self.fc = nn.Sequential(
            nn.LazyLinear(out_features=1024),
            nn.ReLU(),
            nn.Dropout(),
            nn.LazyLinear(out_features=n_outputs)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x