import numpy as np

def moe_load_balancing_loss(gate_logits, num_experts, alpha=0.01):
    """
    Compute the load balancing auxiliary loss for a Mixture of Experts layer.
    
    Args:
        gate_logits: numpy array of shape (num_tokens, num_experts), raw gating scores
        num_experts: int, number of experts
        alpha: float, scaling coefficient for the loss
    
    Returns:
        float: load balancing loss rounded to 4 decimal places
    """
    logits = gate_logits - np.max(gate_logits, axis = 1, keepdims = True)
    exp = np.exp(logits)
    probs = exp / np.sum(exp, axis = 1, keepdims = True)

    assignments = np.argmax(probs, axis = -1)
    f = np.bincount(assignments, minlength = num_experts) / len(assignments)
    p = np.mean(probs, axis = 0)
    loss = alpha * num_experts * np.sum(f * p)
    return round(float(loss), 4)