import torch.nn as nn
import torch


class TextExtractor(nn.Module):
    def __init__(self):
        super(TextExtractor, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, (8, 8), stride=2)
        self.conv2 = nn.Conv2d(32, 64, (5, 5), stride=2)
        self.conv3 = nn.Conv2d(64, 128, (3, 3), stride=2)
        self.conv4 = nn.Conv2d(128, 256, (2, 2), stride=2)

        self.convt1 = nn.ConvTranspose2d(256, 128, (4, 4), stride=2)
        self.convt2 = nn.ConvTranspose2d(128, 64, (2, 2), stride=2)
        self.convt3 = nn.ConvTranspose2d(64, 64, (2, 2), stride=2)
        self.convt4 = nn.ConvTranspose2d(64, 16, (1, 1), stride=2)
        self.convt5 = nn.ConvTranspose2d(16, 8, (2, 2))
        self.convt6 = nn.ConvTranspose2d(8, 1, (1, 1))

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))

        x = torch.relu(self.convt1(x))
        x = torch.relu(self.convt2(x))
        x = torch.relu(self.convt3(x))
        x = torch.relu(self.convt4(x))
        x = torch.relu(self.convt5(x))
        x = torch.sigmoid(self.convt6(x))

        return x


class BGRestoreNet(nn.Module):
    def __init__(self):
        super(BGRestoreNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, (8, 8), stride=2)
        self.conv2 = nn.Conv2d(32, 64, (5, 5), stride=2)
        self.conv3 = nn.Conv2d(64, 128, (3, 3), stride=2)
        self.conv4 = nn.Conv2d(128, 256, (2, 2), stride=2)

        self.convt1 = nn.ConvTranspose2d(256, 128, (4, 4), stride=2)
        self.convt2 = nn.ConvTranspose2d(128, 64, (2, 2), stride=2)
        self.convt3 = nn.ConvTranspose2d(64, 64, (2, 2), stride=2)
        self.convt4 = nn.ConvTranspose2d(64, 16, (1, 1), stride=2)
        self.convt5 = nn.ConvTranspose2d(16, 8, (2, 2))
        self.convt6 = nn.ConvTranspose2d(8, 3, (1, 1))

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))

        x = torch.relu(self.convt1(x))
        x = torch.relu(self.convt2(x))
        x = torch.relu(self.convt3(x))
        x = torch.relu(self.convt4(x))
        x = torch.relu(self.convt5(x))
        x = torch.sigmoid(self.convt6(x))

        return x


if __name__ == '__main__':
    # test
    from pytorch_model_summary import summary

    device = torch.device("cuda" if torch.cuda.is_available()
                          else "cpu")  # PyTorch v0.4.0
    model = TextExtractor().to(device)

    print(summary(model, torch.zeros((1, 1, 256, 256)), show_input=True))
