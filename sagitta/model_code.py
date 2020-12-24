"""
Pytorch Sagitta Model Code
Paper: Untangling the Galaxy III: Photometric Search for Pre-main Sequence Stars with Deep Learning
Authors: Aidan McBride, Ryan Lingg, Marina Kounkel, Kevin Covey, and Brian Hutchinson
"""

#--------------- External Imports ---------------#
import torch

#--------------- Sagitta Convolutional Neural Network ---------------#
class Sagitta(torch.nn.Module):
    """
    Sagitta model class
    """

    def __init__(self, connectShape=60, drop_p=0):
        """
        Model constructor
        """
        super().__init__()
        self.feats = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=1, out_channels=32,
                            kernel_size=5, stride=1, padding=2),
            torch.nn.MaxPool1d(kernel_size=2, stride=2, padding=0),
            torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm1d(num_features=32),
            torch.nn.Conv1d(in_channels=32, out_channels=64,
                            kernel_size=3, stride=1, padding=2),
            torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm1d(num_features=64),
            torch.nn.Conv1d(in_channels=64, out_channels=64,
                            kernel_size=3, stride=1, padding=2),
            torch.nn.MaxPool1d(kernel_size=2, stride=2, padding=0),
            torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm1d(num_features=64),
            torch.nn.Conv1d(in_channels=64, out_channels=128,
                            kernel_size=3, stride=1, padding=2),
            torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm1d(num_features=128)
        )
        self.dropout = torch.nn.Dropout(p=drop_p)
        self.classifier = torch.nn.Conv1d(in_channels=128, out_channels=10,
                                          kernel_size=1, stride=1, padding=0)
        self.fc1 = torch.nn.Linear(in_features=connectShape, out_features=512)
        self.fc2 = torch.nn.Linear(in_features=512, out_features=512)
        self.fc3 = torch.nn.Linear(in_features=512, out_features=1)

    def forward(self, inputs):
        """
        Forward formula for data through the model
        """
        out = self.feats(inputs)
        out = self.dropout(out)
        out = self.classifier(out)
        out = out.view(out.size(0), -1)
        out = torch.nn.functional.relu(self.fc1(out))
        out = torch.nn.functional.relu(self.fc2(out))
        out = self.dropout(out)
        out = self.fc3(out)
        return out
