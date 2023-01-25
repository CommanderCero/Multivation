import torch
import torch.nn as nn

def create_conv(channels, activation=nn.ReLU, kernel_size=(3,3), stride=2, padding=1):
    layers = []
    for i in range(len(channels) - 1):
        layers.append(nn.Conv2d(
            in_channels=channels[i],
            out_channels=channels[i+1],
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        ))
        layers.append(activation())
    return nn.Sequential(*layers)

def create_feedforward(sizes, activation=nn.ReLU): 
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i+1]))
        if i < len(sizes) - 2:
            layers.append(activation())
    return nn.Sequential(*layers)

def sample_weighting_vector(shape: tuple, device=None) -> torch.FloatTensor:
    """
    Uniformly samples a vector of a given shape, such that the sum of all elements in the last axis equals 1.
    
    Randomly sampling values and then normalizing them results in an uneven distribution,
    which tends to assign equal weightings accross all elements.
    This implementation avoids this issue.
    """
    # Implementation is based on this stackoverflow post:
    # https://stats.stackexchange.com/questions/14059/generate-uniformly-distributed-weights-that-sum-to-unity
    weights = torch.rand(shape, dtype=torch.float32, device=device)
    weights = -torch.log(weights)
    weights = weights / weights.sum(axis=-1, keepdim=True)
    return weights

def copy_parameters(source_model: nn.Module, target_model: nn.Module):
    source_params = dict(source_model.named_parameters())
    for name, target_param in target_model.named_parameters():
        assert name in source_params, "Cannot copy model parameters because '{name}' is missing in the source model."
        target_param.data.copy_(source_params[name])
        
def polyak_averaging(local_model: nn.Module, target_model: nn.Module, tau: float):
    """
    Interpolates the models weights towards the target weights using polyak averaging.
    new_weight = tau * target_param + (1 - tau) * model_param
    """
    local_params = dict(local_model.named_parameters())
    for name, target_param in target_model.named_parameters():
        assert name in local_params, "Cannot interpolate target model parameters because '{name}' is missing in the local model."
        target_param.data.mul_(tau)
        target_param.data.add_((1-tau) * local_params[name].data)