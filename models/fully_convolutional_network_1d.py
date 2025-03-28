import torch
from torch import nn

from models.nn_model import NNModel

class FullyConvolutionalNetwork1D(NNModel):

    # def __init__(self, train_dataloader, test_dataloader, metadata, model_name = 'FullyConvolutionalNetwork1D', random_state=42) -> None:
    #     super().__init__(train_dataloader, test_dataloader, metadata, model_name, random_state)

    def __init__(self, dataset_name, train_dataset, test_dataset, metadata, model_name = 'FullyConvolutionalNetwork1D', random_state=42, device='cuda', is_multimodal=False) -> None:
        super().__init__(dataset_name, train_dataset, test_dataset, metadata, model_name, random_state, device, is_multimodal)

        self.network = nn.Sequential(
            nn.Conv1d(in_channels=self.input_shape[1], out_channels=128, kernel_size=8),
            nn.ReLU(),
            nn.BatchNorm1d(128),

            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5),
            nn.ReLU(),
            nn.BatchNorm1d(256),

            nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm1d(128),

            nn.AdaptiveAvgPool1d(output_size=1),
            nn.Flatten(),
            nn.Linear(in_features= 128, out_features=self.classes_shape[1])

        )


    def forward(self, x):
        return self.network(x)
