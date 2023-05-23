from torch import nn
import torch.nn.functional as F

def make_model(args):
    return CNN_CIFAR_dropout(args)

class CNN_CIFAR_dropout(nn.Module):
    """Model Used by the paper introducing FedAvg"""

    def __init__(self, args):
        super(CNN_CIFAR_dropout, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=32, kernel_size=(3, 3)
        )
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=(3, 3)
        )
        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=(3, 3)
        )

        self.fc1 = nn.Linear(4 * 4 * 64, 64)
        self.fc2 = nn.Linear(64, 10)

        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x, is_feat=False, preact=False):
        f1_pre = self.conv1(x)
        f1 = F.relu(f1_pre)
        x = f1
        x = F.max_pool2d(x, 2, 2)
        x = self.dropout(x)

        f2_pre = self.conv2(x)
        f2 = F.relu(f2_pre)
        x = f2
        x = F.max_pool2d(x, 2, 2)
        x = self.dropout(x)

        f3_pre = self.conv3(x)
        f3 = f3_pre
        x = f3
        x = self.dropout(x)
        x = x.view(-1, 4 * 4 * 64)

        f4_pre = self.fc1(x)
        f4 = F.relu(f4_pre)
        x = f4

        x = self.fc2(x)

        if is_feat:
            if preact:
                return [f1_pre, f2_pre, f3_pre, f4_pre], F.log_softmax(x, dim=1)
            else:
                return [f1, f2, f3, f4], F.log_softmax(x, dim=1)
        else:
            return F.log_softmax(x, dim=1)