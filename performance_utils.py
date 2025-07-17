"""
Performance monitoring utilities for DGS-GPT
"""
import time
import torch
import psutil
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Container for performance metrics"""
    timestamp: datetime = field(default_factory=datetime.now)
    step: int = 0
    loss: float = 0.0
    learning_rate: float = 0.0
    tokens_per_second: float = 0.0
    memory_allocated_gb: float = 0.0
    memory_reserved_gb: float = 0.0
    cpu_percent: float = 0.0
    gpu_utilization: Optional[float] = None
    
class PerformanceMonitor:
    """Monitor training and inference performance"""
    
    def __init__(self, device: torch.device):
        self.device = device
        self.metrics_history: List[PerformanceMetrics] = []
        self.start_time = time.time()
        self.step_start_time = time.time()
        self.tokens_processed = 0
        
    def start_step(self):
        """Mark the start of a training/inference step"""
        self.step_start_time = time.time()
        
    def end_step(self, step: int, loss: float, learning_rate: float, tokens_in_batch: int):
        """Record metrics at the end of a step"""
        step_time = time.time() - self.step_start_time
        tokens_per_second = tokens_in_batch / step_time if step_time > 0 else 0
        
        metrics = PerformanceMetrics(
            step=step,
            loss=loss,
            learning_rate=learning_rate,
            tokens_per_second=tokens_per_second
        )
        
        # Get memory metrics
        if self.device.type == "cuda":
            metrics.memory_allocated_gb = torch.cuda.memory_allocated(self.device) / 1024**3
            metrics.memory_reserved_gb = torch.cuda.memory_reserved(self.device) / 1024**3
            
            # Try to get GPU utilization if nvidia-ml-py is available
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(self.device.index or 0)
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                metrics.gpu_utilization = utilization.gpu
            except ImportError:
                metrics.gpu_utilization = None
            except Exception as e:
                logger.debug(f"Could not get GPU utilization: {e}")
                metrics.gpu_utilization = None
        
        # Get CPU metrics
        metrics.cpu_percent = psutil.cpu_percent()
        
        self.metrics_history.append(metrics)
        self.tokens_processed += tokens_in_batch
        
        return metrics
    
    def get_average_metrics(self, last_n_steps: Optional[int] = None) -> Dict[str, float]:
        """Get average metrics over the last N steps"""
        if not self.metrics_history:
            return {}
            
        recent_metrics = self.metrics_history[-last_n_steps:] if last_n_steps else self.metrics_history
        
        if not recent_metrics:
            return {}
            
        return {
            'avg_loss': sum(m.loss for m in recent_metrics) / len(recent_metrics),
            'avg_tokens_per_second': sum(m.tokens_per_second for m in recent_metrics) / len(recent_metrics),
            'avg_memory_allocated_gb': sum(m.memory_allocated_gb for m in recent_metrics) / len(recent_metrics),
            'avg_cpu_percent': sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics),
            'total_steps': len(recent_metrics),
            'total_time_hours': (time.time() - self.start_time) / 3600
        }
    
    def log_summary(self, last_n_steps: int = 100):
        """Log a performance summary"""
        avg_metrics = self.get_average_metrics(last_n_steps)
        
        if avg_metrics:
            logger.info("=== Performance Summary ===")
            logger.info(f"Average Loss (last {last_n_steps} steps): {avg_metrics['avg_loss']:.4f}")
            logger.info(f"Average Tokens/sec: {avg_metrics['avg_tokens_per_second']:.1f}")
            logger.info(f"Average GPU Memory: {avg_metrics['avg_memory_allocated_gb']:.1f} GB")
            logger.info(f"Average CPU Usage: {avg_metrics['avg_cpu_percent']:.1f}%")
            logger.info(f"Total Training Time: {avg_metrics['total_time_hours']:.2f} hours")
            logger.info(f"Total Tokens Processed: {self.tokens_processed:,}")
    
    def detect_performance_issues(self) -> List[str]:
        """Detect potential performance issues"""
        issues = []
        
        if len(self.metrics_history) < 10:
            return issues
            
        recent_metrics = self.metrics_history[-10:]
        
        # Check for low GPU utilization
        gpu_utils = [m.gpu_utilization for m in recent_metrics if m.gpu_utilization is not None]
        if gpu_utils and sum(gpu_utils) / len(gpu_utils) < 50:
            issues.append("Low GPU utilization detected (< 50%)")
        
        # Check for memory issues
        memory_usage = [m.memory_allocated_gb for m in recent_metrics]
        if memory_usage and max(memory_usage) > 10:  # More than 10GB
            issues.append("High GPU memory usage detected")
        
        # Check for slow token processing
        token_rates = [m.tokens_per_second for m in recent_metrics]
        if token_rates and sum(token_rates) / len(token_rates) < 100:
            issues.append("Low token processing rate (< 100 tokens/sec)")
        
        # Check for high CPU usage (might indicate CPU bottleneck)
        cpu_usage = [m.cpu_percent for m in recent_metrics]
        if cpu_usage and sum(cpu_usage) / len(cpu_usage) > 90:
            issues.append("High CPU usage detected (> 90%)")
        
        return issues
    
    def export_metrics(self, filepath: str):
        """Export metrics to a file"""
        try:
            import pandas as pd
            
            # Convert metrics to dictionary format
            data = []
            for m in self.metrics_history:
                data.append({
                    'timestamp': m.timestamp.isoformat(),
                    'step': m.step,
                    'loss': m.loss,
                    'learning_rate': m.learning_rate,
                    'tokens_per_second': m.tokens_per_second,
                    'memory_allocated_gb': m.memory_allocated_gb,
                    'memory_reserved_gb': m.memory_reserved_gb,
                    'cpu_percent': m.cpu_percent,
                    'gpu_utilization': m.gpu_utilization
                })
            
            df = pd.DataFrame(data)
            
            if filepath.endswith('.csv'):
                df.to_csv(filepath, index=False)
            elif filepath.endswith('.json'):
                df.to_json(filepath, orient='records', date_format='iso')
            else:
                # Default to CSV
                df.to_csv(filepath + '.csv', index=False)
                
            logger.info(f"Performance metrics exported to {filepath}")
            
        except ImportError:
            logger.warning("pandas not available, cannot export metrics to file")
        except Exception as e:
            logger.error(f"Error exporting metrics: {e}")

# Context manager for timing operations
class Timer:
    """Context manager for timing operations"""
    
    def __init__(self, name: str = "Operation", logger=None):
        self.name = name
        self.logger = logger or logging.getLogger(__name__)
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        self.logger.info(f"{self.name} completed in {duration:.2f} seconds")
    
    @property
    def elapsed_time(self) -> float:
        """Get elapsed time"""
        if self.start_time is None:
            return 0.0
        end = self.end_time or time.time()
        return end - self.start_time
