"""
Improved logging system for DGS-GPT
"""
import logging
import sys
import os
from datetime import datetime
from pathlib import Path

# Emoji fallback mapping for Windows console compatibility
EMOJI_FALLBACK = {
    '🚀': '[ROCKET]',
    '⚡': '[LIGHTNING]', 
    '🎯': '[TARGET]',
    '💾': '[SAVE]',
    '🧹': '[CLEAN]',
    '✅': '[CHECK]',
    '❌': '[CROSS]',
    '🔥': '[FIRE]',
    '💡': '[BULB]',
    '📊': '[CHART]',
    '👋': '[WAVE]'
}

def safe_emoji_text(text):
    """Replace emoji characters with safe alternatives on Windows"""
    if os.name == 'nt':  # Windows
        for emoji, fallback in EMOJI_FALLBACK.items():
            text = text.replace(emoji, fallback)
    return text

class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for different log levels"""
    
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record):
        # Replace emoji characters with safe alternatives on Windows
        record.msg = safe_emoji_text(str(record.msg))
        
        log_color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{log_color}{record.levelname}{self.RESET}"
        return super().format(record)

def setup_logging(log_level=logging.INFO, log_file=None):
    """Setup logging configuration"""
    
    # Create formatter
    formatter = ColoredFormatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Setup console handler with UTF-8 encoding support
    try:
        # Try to create a UTF-8 compatible console handler
        if os.name == 'nt':  # Windows
            # Force UTF-8 encoding on Windows console
            import io
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        
        # Set encoding to UTF-8 to handle emoji characters
        if hasattr(console_handler.stream, 'reconfigure'):
            try:
                console_handler.stream.reconfigure(encoding='utf-8', errors='replace')
            except Exception:
                pass  # Fallback if reconfigure is not available
                
    except Exception:
        # Fallback to basic handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.addHandler(console_handler)
    
    # Setup file handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
    
    return root_logger

# Create module-level logger
logger = logging.getLogger(__name__)
