import torch
import torch.optim as optim
from typing import Dict, Any, Optional


def create_lr_scheduler(optimizer: torch.optim.Optimizer, 
                       scheduler_config: Dict[str, Any], 
                       num_epochs: int) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """
    Create a learning rate scheduler based on configuration.
    
    Args:
        optimizer: PyTorch optimizer
        scheduler_config: Configuration dictionary for the scheduler
        num_epochs: Total number of training epochs
        
    Returns:
        Learning rate scheduler or None if disabled
    """
    if not scheduler_config.get('enabled', False):
        return None
    
    scheduler_type = scheduler_config.get('type', 'cosine').lower()
    
    if scheduler_type == 'cosine':
        T_max = scheduler_config.get('cosine_T_max', num_epochs)
        eta_min = scheduler_config.get('cosine_eta_min', 0.000001)
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=T_max,
            eta_min=eta_min
        )
    
    elif scheduler_type == 'step':
        step_size = scheduler_config.get('step_size', 20)
        gamma = scheduler_config.get('step_gamma', 0.5)
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma
        )
    
    elif scheduler_type == 'exponential':
        gamma = scheduler_config.get('exp_gamma', 0.95)
        return optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=gamma
        )
    
    elif scheduler_type == 'plateau':
        mode = scheduler_config.get('plateau_mode', 'min')
        factor = scheduler_config.get('plateau_factor', 0.5)
        patience = scheduler_config.get('plateau_patience', 10)
        threshold = scheduler_config.get('plateau_threshold', 0.0001)
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=mode,
            factor=factor,
            patience=patience,
            threshold=threshold,
            verbose=True
        )
    
    else:
        print(f"Warning: Unknown scheduler type '{scheduler_type}'. No scheduler will be used.")
        return None


def step_scheduler(scheduler: Optional[torch.optim.lr_scheduler._LRScheduler], 
                  metric: Optional[float] = None) -> None:
    """
    Step the learning rate scheduler.
    
    Args:
        scheduler: Learning rate scheduler (can be None)
        metric: Validation metric for ReduceLROnPlateau scheduler
    """
    if scheduler is None:
        return
    
    if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
        if metric is not None:
            scheduler.step(metric)
        else:
            print("Warning: ReduceLROnPlateau scheduler requires a metric but none provided.")
    else:
        scheduler.step()


def get_current_lr(optimizer: torch.optim.Optimizer) -> float:
    """
    Get the current learning rate from the optimizer.
    
    Args:
        optimizer: PyTorch optimizer
        
    Returns:
        Current learning rate
    """
    return optimizer.param_groups[0]['lr'] 