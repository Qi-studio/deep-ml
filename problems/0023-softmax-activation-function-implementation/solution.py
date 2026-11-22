import torch
import torch.nn.functional as F

def softmax(scores: list[float]) -> list[float]:
    """
    Compute the softmax activation function using PyTorch's built-in API.
    Input:
      - scores: list of floats (logits)
    Returns:
      - list of floats representing the softmax probabilities.
    """
    # Your implementation here
    scores_tensor = torch.as_tensor(scores, dtype = torch.float64)
    # max_value = torch.max(scores_tensor)
    # exp_values = torch.exp(scores_tensor - max_value)
    # sum_exp = torch.sum(exp_values)
    # softmax = exp_values / sum_exp
    # return softmax.tolist()
    return F.softmax(scores_tensor).tolist()
