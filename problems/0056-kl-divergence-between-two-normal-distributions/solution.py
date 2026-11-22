import torch
from torch.distributions import Normal, kl_divergence

def kl_divergence_normal(mu_p, sigma_p, mu_q, sigma_q) -> torch.Tensor:
    """
    Compute the KL divergence between two normal distributions P and Q.
    
    Args:
        mu_p: Mean of distribution P
        sigma_p: Standard deviation of distribution P
        mu_q: Mean of distribution Q
        sigma_q: Standard deviation of distribution Q
    
    Returns:
        torch.Tensor: KL divergence KL(P || Q)
    """
    p = Normal(mu_p, sigma_p)
    q = Normal(mu_q, sigma_q)
    return kl_divergence(p, q)