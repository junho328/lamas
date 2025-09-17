import os
import numpy as np
from typing import Optional, Dict, Any
from tensorboardX import SummaryWriter

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None


class Logger:
    """Unified logging interface supporting both TensorBoard and wandb."""
    
    def __init__(self, 
                 log_dir: str,
                 use_wandb: bool = False,
                 wandb_project: str = "marft-math",
                 wandb_entity: Optional[str] = None,
                 wandb_run_name: Optional[str] = None,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize logger with TensorBoard and/or wandb support.
        
        Args:
            log_dir: Directory for TensorBoard logs
            use_wandb: Whether to use wandb logging
            wandb_project: wandb project name
            wandb_entity: wandb entity name
            wandb_run_name: wandb run name
            config: Configuration dictionary to log to wandb
        """
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.log_dir = log_dir
        
        # Initialize TensorBoard writer
        self.tb_writer = SummaryWriter(log_dir)
        
        # Initialize wandb if requested and available
        if self.use_wandb:
            if not WANDB_AVAILABLE:
                print("Warning: wandb not available, falling back to TensorBoard only")
                self.use_wandb = False
            else:
                wandb_config = config or {}
                wandb.init(
                    project=wandb_project,
                    entity=wandb_entity,
                    name=wandb_run_name,
                    config=wandb_config,
                    dir=log_dir
                )
                print(f"Initialized wandb logging for project: {wandb_project}")
    
    def add_scalar(self, tag: str, scalar_value: float, global_step: int):
        """Log a scalar value."""
        self.tb_writer.add_scalar(tag, scalar_value, global_step)
        
        if self.use_wandb:
            wandb.log({tag: scalar_value}, step=global_step)
    
    def add_scalars(self, main_tag: str, tag_scalar_dict: Dict[str, float], global_step: int):
        """Log multiple scalar values."""
        self.tb_writer.add_scalars(main_tag, tag_scalar_dict, global_step)
        
        if self.use_wandb:
            # Flatten the nested dictionary for wandb
            wandb_dict = {f"{main_tag}/{k}": v for k, v in tag_scalar_dict.items()}
            wandb.log(wandb_dict, step=global_step)
    
    def add_histogram(self, tag: str, values: np.ndarray, global_step: int):
        """Log a histogram."""
        self.tb_writer.add_histogram(tag, values, global_step)
        
        if self.use_wandb:
            wandb.log({tag: wandb.Histogram(values)}, step=global_step)
    
    def add_text(self, tag: str, text_string: str, global_step: int):
        """Log text."""
        self.tb_writer.add_text(tag, text_string, global_step)
        
        if self.use_wandb:
            wandb.log({tag: wandb.Html(text_string)}, step=global_step)
    
    def add_figure(self, tag: str, figure, global_step: int):
        """Log a matplotlib figure."""
        self.tb_writer.add_figure(tag, figure, global_step)
        
        if self.use_wandb:
            wandb.log({tag: wandb.Image(figure)}, step=global_step)
    
    def log_metrics(self, metrics: Dict[str, float], step: int):
        """Log a dictionary of metrics."""
        for key, value in metrics.items():
            self.add_scalar(key, value, step)
    
    def watch_model(self, model, log: str = "gradients", log_freq: int = 100):
        """Watch model parameters and gradients (wandb only)."""
        if self.use_wandb:
            wandb.watch(model, log=log, log_freq=log_freq)
    
    def log_artifact(self, artifact_path: str, name: str, artifact_type: str = "model"):
        """Log an artifact (wandb only)."""
        if self.use_wandb:
            artifact = wandb.Artifact(name, type=artifact_type)
            artifact.add_file(artifact_path)
            wandb.log_artifact(artifact)
    
    def export_scalars_to_json(self, filepath: str):
        """Export scalars to JSON (TensorBoard only)."""
        self.tb_writer.export_scalars_to_json(filepath)
    
    def close(self):
        """Close all loggers."""
        self.tb_writer.close()
        
        if self.use_wandb:
            wandb.finish()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
