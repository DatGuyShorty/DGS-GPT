"""
Enhanced logging utilities for DGS-GPT.

This module provides structured logging with different levels,
file rotation, and formatted output for better debugging and monitoring.
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional
import traceback

def setup_logging(
    log_level: int = logging.INFO,
    log_file: Optional[str] = None,
    max_file_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    console_output: bool = True
) -> logging.Logger:
    """
    Set up comprehensive logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (optional)
        max_file_size: Maximum log file size before rotation
        backup_count: Number of backup files to keep
        console_output: Whether to output to console
        
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger("dgs_gpt")
    logger.setLevel(log_level)
    
    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler with rotation
    if log_file:
        try:
            # Ensure log directory exists
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=max_file_size,
                backupCount=backup_count,
                encoding='utf-8'
            )
            file_handler.setLevel(log_level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            
            logger.info(f"Logging to file: {log_file}")
            
        except Exception as e:
            logger.warning(f"Could not set up file logging: {e}")
    
    # Add exception handler
    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        
        logger.critical(
            "Uncaught exception",
            exc_info=(exc_type, exc_value, exc_traceback)
        )
    
    sys.excepthook = handle_exception
    
    return logger

def log_system_info(logger: logging.Logger):
    """Log system information for debugging purposes."""
    try:
        import torch
        import platform
        import sys
        
        logger.info("=== SYSTEM INFORMATION ===")
        logger.info(f"Platform: {platform.platform()}")
        logger.info(f"Python: {sys.version}")
        logger.info(f"PyTorch: {torch.__version__}")
        
        if torch.cuda.is_available():
            logger.info(f"CUDA: {torch.version.cuda}")
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f}GB")
        else:
            logger.info("CUDA: Not available")
        
        logger.info("=== END SYSTEM INFO ===")
        
    except Exception as e:
        logger.error(f"Error logging system info: {e}")

def log_config_info(logger: logging.Logger, config):
    """Log configuration information."""
    try:
        logger.info("=== MODEL CONFIGURATION ===")
        logger.info(f"Model Type: {config.model_type}")
        logger.info(f"VRAM Profile: {config.vram_profile}")
        logger.info(f"Architecture: {config.n_layer} layers, {config.n_head} heads, {config.n_embd} embedding")
        logger.info(f"Attention: {config.attention_type}")
        
        if config.use_moe:
            logger.info(f"MoE: {config.moe_num_experts} experts, top-{config.moe_k}")
        else:
            logger.info("MoE: Disabled")
            
        logger.info(f"Training: batch={config.batch_size}, lr={config.learning_rate}, wd={config.weight_decay}")
        logger.info(f"Context: {config.block_size} tokens")
        logger.info("=== END CONFIG INFO ===")
        
    except Exception as e:
        logger.error(f"Error logging config info: {e}")

# Create default logger instance
logger = setup_logging()

class TrainingLogger:
    """Specialized logger for training metrics and progress."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.step_count = 0
        self.best_loss = float('inf')
        
    def log_step(self, step: int, loss: float, lr: float, **kwargs):
        """Log training step information."""
        self.step_count = step
        
        # Track best loss
        if loss < self.best_loss:
            self.best_loss = loss
            is_best = " (BEST)"
        else:
            is_best = ""
        
        # Log every 10 steps or if it's the best
        if step % 10 == 0 or is_best:
            self.logger.info(
                f"Step {step}: loss={loss:.4f}, lr={lr:.6f}{is_best}"
            )
            
        # Log additional metrics if provided
        for key, value in kwargs.items():
            if step % 50 == 0:  # Log detailed metrics less frequently
                self.logger.debug(f"Step {step} {key}: {value}")
    
    def log_epoch(self, epoch: int, avg_loss: float, **kwargs):
        """Log epoch summary."""
        self.logger.info(f"Epoch {epoch} complete: avg_loss={avg_loss:.4f}")
        
        for key, value in kwargs.items():
            self.logger.info(f"Epoch {epoch} {key}: {value}")
    
    def log_evaluation(self, eval_loss: float, perplexity: float, **kwargs):
        """Log evaluation results."""
        self.logger.info(f"Evaluation: loss={eval_loss:.4f}, perplexity={perplexity:.2f}")
        
        for key, value in kwargs.items():
            self.logger.info(f"Evaluation {key}: {value}")

class ErrorHandler:
    """Centralized error handling and logging."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        
    def handle_error(self, error: Exception, context: str = "", critical: bool = False):
        """Handle and log errors with context."""
        error_msg = f"Error in {context}: {str(error)}"
        
        if critical:
            self.logger.critical(error_msg)
            self.logger.critical(traceback.format_exc())
        else:
            self.logger.error(error_msg)
            self.logger.debug(traceback.format_exc())
            
    def handle_warning(self, message: str, context: str = ""):
        """Handle and log warnings."""
        warning_msg = f"Warning in {context}: {message}" if context else f"Warning: {message}"
        self.logger.warning(warning_msg)
        
    def handle_info(self, message: str, context: str = ""):
        """Handle and log info messages."""
        info_msg = f"{context}: {message}" if context else message
        self.logger.info(info_msg)

# Create global error handler
error_handler = ErrorHandler(logger)
