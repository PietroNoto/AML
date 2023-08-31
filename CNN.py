from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gym.spaces import Box
from models.mobilenet import get_model
import torch

class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: Box):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.model=get_model(num_classes=11, sample_size=n_input_channels, width_mult=2.)
        checkpoint = torch.load("/mnt/c/Users/utente/Downloads/jester_mobilenet_2.0x_RGB_16_best.pth",map_location=torch.device("cpu"))
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        #settare train sull'ultimo layer
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.model(observations)