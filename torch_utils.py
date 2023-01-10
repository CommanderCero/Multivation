import torch
import torch.nn as nn

def create_feedforward(sizes, activation=nn.ReLU): 
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i+1]))
        if i < len(sizes) - 2:
            layers.append(activation())
    return nn.Sequential(*layers)

def sample_weighting_vector(size: int) -> torch.FloatTensor:
    """
    Uniformly samples a vector such that the sum of all elements equals 1.
    
    Randomly sampling values and then normalizing them results in an uneven distribution,
    which tends to assign equal weightings accross all elements.
    This implementation avoids this issue.
    """
    # Implementation is based on this stackoverflow post:
    # https://stats.stackexchange.com/questions/14059/generate-uniformly-distributed-weights-that-sum-to-unity
    values = torch.rand(size, dtype=torch.float32)
    values = -torch.log(values)
    weights = values / values.sum()
    return weights

def copy_parameters(source_model: nn.Module, target_model: nn.Module):
    source_params = dict(source_model.named_parameters())
    for name, target_param in target_model.named_parameters():
        assert name in source_params, "Cannot copy model parameters because '{name}' is missing in the source model."
        target_param.data.copy_(source_params[name])
        
def polyak_interpolation(model: nn.Module, target_model: nn.Module, tau: float):
    """
    Interpolates the models weights towards the target weights using polyak averaging.
    new_weight = tau * target_param + (1 - tau) * model_param
    """
    target_params = dict(target_model.named_parameters())
    for name, param in model.named_parameters():
        assert name in target_params, "Cannot interpolate model parameters because '{name}' is missing in the target model."
        new_weight = tau * target_params[name].data + (1 - tau) * param.data
        param.data.copy_(new_weight)