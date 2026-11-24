import torch

def moe_load_balancing_loss(gate_logits, num_experts: int, alpha: float = 0.01) -> float:
    """
    Compute the load balancing auxiliary loss for a Mixture of Experts layer.
    
    Args:
        gate_logits: torch.Tensor of shape (num_tokens, num_experts), raw gating scores
        num_experts: int, number of experts
        alpha: float, scaling coefficient for the loss
    
    Returns:
        float: load balancing loss rounded to 4 decimal places
    """
    gate_logits = gate_logits - torch.max(gate_logits, dim = -1, keepdims = True).values
    exp = torch.exp(gate_logits)
    probs = exp / torch.sum(exp, dim = -1, keepdims = True)

    assignments = torch.argmax(probs, dim = -1)
    f = torch.bincount(assignments, minlength = num_experts).float() / len(assignments)
    p = torch.mean(probs, dim = 0)
    loss = alpha * num_experts * torch.sum(f * p)
    return round(loss.item(), 4)