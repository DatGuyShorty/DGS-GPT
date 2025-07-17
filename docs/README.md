# DGS-GPT 🤖

A modern, feature-rich GPT implementation with advanced training capabilities, hyperparameter optimization, and an intuitive GUI interface.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/framework-PyTorch-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## ✨ Features

### 🧠 Advanced Model Architecture
- **Multi-Head Attention** with configurable heads and embedding dimensions
- **Grouped Query Attention** for improved efficiency
- **Mixture of Experts (MoE)** for scalable model capacity
- **Configurable transformer blocks** with LayerNorm and dropout
- **Character-level tokenization** with custom vocabulary

### 🎯 Hyperparameter Optimization
- **Optuna-powered optimization** with pruning for efficient search
- **Automatic VRAM management** to prevent out-of-memory errors
- **Persistent study storage** with SQLite database
- **Best parameters auto-save** to JSON for reproducibility

### 🖥️ Graphical User Interface
- **Three-tab interface**: Text Generation, Training, and Hyperparameter Optimization
- **Real-time training visualization** with loss plotting
- **Live hyperparameter display** showing current model configuration
- **Progress tracking** with progress bars and status updates
- **Model save/load functionality** through file dialogs

### 🚀 Training Features
- **Mixed precision training** with automatic scaling (CUDA)
- **Gradient accumulation** and weight decay
- **Real-time loss monitoring** with customizable plot intervals
- **Stop/resume functionality** with threading support
- **Memory-efficient batching** with configurable batch sizes

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-compatible GPU (recommended) or CPU
- Git

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/DatGuyShorty/DGS-GPT.git
   cd DGS-GPT
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare your dataset:**
   - Place your text data in `vocab.txt`
   - The model will automatically create character-level vocabulary
   - Sample datasets can be any plain text file

## 🚀 Quick Start

### Using the GUI (Recommended)
```bash
python gui.py
```

This launches the full-featured GUI with three main tabs:
- **Text Generation**: Generate text from prompts
- **Training**: Train the model with real-time visualization
- **Hyperparameter Optimization**: Auto-tune model parameters

### Command Line Usage
```bash
python ShitGPT.py
```

## 📖 Usage Guide

### 1. Text Generation
1. Open the GUI and navigate to the "Text Generation" tab
2. Enter your prompt in the text area
3. Click "Generate Text" to produce AI-generated content
4. Generated text appears in the output area below

### 2. Model Training
1. Go to the "Training" tab
2. Set training parameters:
   - **Iterations**: Number of training steps
   - **Plot Step**: How often to update the loss plot
3. View current hyperparameters in the display widget
4. Click "Start Training" to begin
5. Monitor progress with the real-time loss plot and progress bar
6. Use "Stop Training" to halt training early if needed

### 3. Hyperparameter Optimization
1. Navigate to the "Hyperparameter Optimization" tab
2. Configure optimization settings:
   - **Number of Trials**: How many parameter combinations to test
   - **Steps per Trial**: Training steps per configuration
3. Click "Start Optimization" to begin automated tuning
4. Monitor progress in the log output
5. Best parameters are automatically saved and applied

### 4. Model Management
- **Save Model**: File → Save Model (or Ctrl+S)
- **Load Model**: File → Load Model (or Ctrl+O)
- Models are saved as `.pth` files with complete state dictionaries

## ⚙️ Configuration

### Model Architecture Parameters
```python
@dataclass
class Config:
    # Model Architecture
    block_size: int = 256        # Sequence length
    n_layer: int = 16           # Number of transformer layers
    n_head: int = 16            # Number of attention heads
    n_embd: int = 1024          # Embedding dimension
    dropout: float = 0.01       # Dropout rate
    
    # Advanced Features
    use_moe: bool = False       # Enable Mixture of Experts
    moe_num_experts: int = 4    # Number of MoE experts
    moe_k: int = 1              # Top-k experts to use
    attention_type: str = "multihead"  # "multihead" or "grouped_query"
    n_query_groups: int = 1     # Query groups for GQA
    
    # Training Parameters
    learning_rate: float = 5e-4  # Learning rate
    weight_decay: float = 1e-5   # L2 regularization
    batch_size: int = 24         # Batch size
    max_iters: int = 1000        # Maximum training iterations
```

### Hyperparameter Optimization Ranges
- **Learning Rate**: 1e-5 to 1e-2 (log scale)
- **Weight Decay**: 1e-6 to 1e-2 (log scale)
- **Embedding Dimension**: [256, 512, 1024]
- **Number of Layers**: Adaptive based on embedding size
- **Attention Heads**: [4, 8, 16]
- **Dropout**: 0.05 to 0.3

## 🏗️ Project Structure

```
DGS-GPT/
├── ShitGPT.py              # Core model implementation
├── gui.py                  # Graphical user interface
├── dataset.py              # Data loading utilities
├── live_training_plot.py   # Real-time plotting utilities
├── requirements.txt        # Python dependencies
├── setup.sh               # Setup script
├── vocab.txt              # Training text data
├── best_hyperparams.json  # Optimized parameters
├── optuna_study.db        # Optimization history
├── gpt_model.pth          # Trained model weights
└── README.md              # This file
```

## 🔧 Advanced Features

### Mixture of Experts (MoE)
Enable MoE for scalable model capacity:
```python
config.use_moe = True
config.moe_num_experts = 8
config.moe_k = 2
```

### Grouped Query Attention
Use GQA for improved efficiency:
```python
config.attention_type = "grouped_query"
config.n_query_groups = 4  # Must divide n_head evenly
```

### Custom Datasets
1. Replace `vocab.txt` with your text data
2. The model automatically builds character-level vocabulary
3. Larger datasets generally produce better results

## 🐛 Troubleshooting

### Common Issues

**CUDA Out of Memory:**
- Reduce `batch_size` in config
- Lower `n_embd` or `n_layer` values
- Close other GPU-intensive applications

**Training Not Starting:**
- Ensure `vocab.txt` exists and contains text
- Check that PyTorch detects your GPU (if available)
- Verify all dependencies are installed

**GUI Not Opening:**
- Install tkinter: `pip install tk`
- On Linux: `sudo apt-get install python3-tk`
- Check display settings for headless systems

### Performance Tips
- Use CUDA-enabled GPU for significant speedup
- Adjust batch size based on available VRAM
- Use mixed precision training (automatic on CUDA)
- Enable MoE for larger model capacity with similar memory usage

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and test thoroughly
4. Commit with descriptive messages: `git commit -m "Add feature"`
5. Push to your fork: `git push origin feature-name`
6. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [PyTorch](https://pytorch.org/) for deep learning
- [Optuna](https://optuna.org/) for hyperparameter optimization
- [Matplotlib](https://matplotlib.org/) for visualization
- Inspired by the GPT architecture and modern transformer implementations

## 📞 Support

If you encounter issues or have questions:
1. Check the [Troubleshooting](#-troubleshooting) section
2. Search existing [Issues](https://github.com/DatGuyShorty/DGS-GPT/issues)
3. Create a new issue with detailed information
4. Include system specs, error messages, and reproduction steps

---

**Happy Training! 🚀**
