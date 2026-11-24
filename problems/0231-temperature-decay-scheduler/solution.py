import torch
import math

def temperature_decay_pytorch(
    schedule_type: str,
    initial_temp: float,
    current_step: int,
    total_steps: int,
    final_temp: float = 0.01,
    decay_rate: float = 0.95
) -> torch.Tensor:
    """
    Compute temperature at current training step using PyTorch.
    
    Same as NumPy version but returns torch.Tensor.
    Useful for integration with PyTorch training loops.
    
    Args:
        schedule_type: 'linear', 'exponential', 'cosine', or 'constant'
        initial_temp: Starting temperature
        current_step: Current training step
        total_steps: Total number of steps
        final_temp: Minimum temperature
        decay_rate: Decay rate for exponential
    
    Returns:
        Temperature as torch.Tensor (scalar)
        
    Hints:
        - Use torch.tensor() to create tensors
        - Use torch.cos() for cosine
        - Use torch.clamp() to enforce minimum
        - Use .item() to extract Python float if needed
    """
    # Your code here
    current_step = torch.tensor(current_step, dtype=torch.float32)
    if schedule_type == "linear":
        return initial_temp + (final_temp - initial_temp) / total_steps * current_step
    elif schedule_type == "exponential":
        return torch.clamp(initial_temp * decay_rate ** current_step, min = final_temp)
    elif schedule_type == "cosine":
        return final_temp + 0.5 * (initial_temp - final_temp) * (1 + torch.cos(torch.pi * current_step / total_steps))
    elif schedule_type == "constant":
        return torch.tensor(initial_temp)