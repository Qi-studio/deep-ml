import torch
import torch.nn.functional as F

def single_neuron_model(features: list[list[float]], labels: list[int], weights: list[float], bias: float) -> tuple[list[float], float]:
    """
    Simulates a single neuron with sigmoid activation for binary classification.
    
    Args:
        features: List of feature vectors (each a list of floats)
        labels: List of true binary labels
        weights: Neuron weights (one per feature)
        bias: Neuron bias term
    
    Returns:
        Tuple of (predicted probabilities rounded to 4 decimal places, MSE rounded to 4 decimal places)
    """
    # Your code here using PyTorch built-ins:
    # - torch.matmul() for linear combination
    # - torch.sigmoid() for activation
    # - torch.nn.functional.mse_loss() for MSE
    feature_tensor = torch.as_tensor(features, dtype = torch.float64)
    labels = torch.as_tensor(labels, dtype = torch.float64)
    weights = torch.as_tensor(weights, dtype = torch.float64)
    # z = feature_tensor @ weights + bias
    z = torch.matmul(feature_tensor, weights) + bias
    activations = torch.sigmoid(z)
    loss = F.mse_loss(activations, labels)
    return [round(i.item(), 4) for i in activations], round(loss.item(), 4)