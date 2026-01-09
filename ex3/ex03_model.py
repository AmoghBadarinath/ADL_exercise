import torch
import torch.nn as nn
import torch.nn.functional as F

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class ShallowCNN(nn.Module):
    def __init__(self, hidden_features=32, num_classes=42, **kwargs):
        super().__init__()
        c_hid1 = hidden_features
        c_hid2 = hidden_features * 2
        c_hid3 = hidden_features * 4

        self.cnn_layers = nn.Sequential(
            nn.Conv2d(1, c_hid1, kernel_size=5, stride=2, padding=4),
            Swish(),
            nn.Conv2d(c_hid1, c_hid2, kernel_size=3, stride=2, padding=1),
            Swish(),
            nn.Conv2d(c_hid2, c_hid3, kernel_size=3, stride=2, padding=1),
            Swish(),
            nn.Conv2d(c_hid3, c_hid3, kernel_size=3, stride=2, padding=1),
            Swish(),
            nn.Conv2d(c_hid3, c_hid3, kernel_size=3, stride=2, padding=1),
            Swish(),
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c_hid3, num_classes)
        )

    def get_logits(self, x):
        # TODO (3.2): Implement classification procedure that outputs the logits across the classes

        # CNN feature extractor
        x = self.cnn_layers(x)

        # Remove spatial dimensions (H×W → 1×1)
        x = F.adaptive_avg_pool2d(x, (1, 1))

        # Pass through fully connected layers to get logits
        logits = self.fc_layers(x)
        return logits

    def forward(self, x, y=None) -> torch.Tensor:
        # TODO (3.2): Implement forward function for (1) EBM, (2) Unconditional JEM, (3) Conditional JEM.
        #  Consider using F.adaptive_avg_pool2d to convert between the 2D features and a linear representation.
        #  (You can also reuse your implementation of 'self.get_logits(x)' if this helps you.)

        logits = self.get_logits(x)
        if y is None:
            # EBM / Unconditional JEM: Calculate logsumexp across all class logits
            return torch.logsumexp(logits, dim=1)
        else:
            # Conditional JEM: Calculate log probability of the given class
            return logits.gather(1, y.view(-1, 1)).squeeze(1)


