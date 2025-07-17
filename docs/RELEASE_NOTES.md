# DGS-GPT v0.1.0-alpha Release Notes

### Added
- **Core GPT Architecture**
  - Transformer-based language model implementation
  - Configurable model parameters (layers, heads, embedding dimensions)
  - Character-level tokenization with custom vocabulary support
  - Mixed precision training for CUDA devices

- **Advanced Attention Mechanisms**
  - Multi-head attention (standard transformer)
  - Grouped Query Attention (GQA) for improved efficiency
  - Configurable attention types per model

- **Mixture of Experts (MoE)**
  - Optional MoE layers for scalable model capacity
  - Configurable number of experts and top-k selection
  - Memory-efficient expert routing

- **Comprehensive GUI Interface**
  - Three-tab design: Text Generation, Training, Hyperparameter Optimization
  - Real-time training loss visualization with matplotlib
  - Live hyperparameter display showing current model configuration
  - Progress bars and status indicators for all operations
  - Model save/load functionality through file dialogs

- **Hyperparameter Optimization**
  - Optuna integration for automated parameter tuning
  - SQLite database storage for optimization history
  - Intelligent parameter space constraints to prevent VRAM overflow
  - Pruning support for efficient optimization
  - Best parameters auto-save to JSON

- **Training Features**
  - Configurable batch size, learning rate, and training iterations
  - Real-time loss callback system for GUI integration
  - Stop/resume functionality with threading
  - Memory cleanup and garbage collection
  - CUDA out-of-memory error handling

- **Cross-Platform Setup**
  - Shell script for Linux/macOS (`setup.sh`)
  - Batch script for Windows (`setup.bat`)
  - Python setup script for universal compatibility (`setup.py`)
  - Virtual environment creation and dependency management
  - Automatic Python version checking (3.8+ required)

- **Dataset Management**
  - Support for custom text datasets via `vocab.txt`
  - Hugging Face datasets integration (`dataset.py`)
  - Automatic vocabulary generation from training data
  - Train/validation/test split management

- **Development Tools**
  - Comprehensive requirements.txt with optimized dependencies
  - .gitignore configured for Python/ML projects
  - Professional README with installation and usage instructions
  - MIT License for open-source distribution

### Technical Details
- **Dependencies**: PyTorch 2.0+, Optuna 3.0+, Matplotlib, NumPy, Datasets
- **Python Version**: 3.8+ required
- **GPU Support**: CUDA-enabled training with automatic fallback to CPU
- **Memory Management**: Automatic garbage collection and CUDA cache clearing
- **Error Handling**: Comprehensive exception handling throughout codebase

### Known Limitations
- Character-level tokenization only (no subword tokenization)
- Single GPU training (no multi-GPU support yet)
- Basic text generation (no advanced sampling strategies)
- Limited dataset preprocessing options

### Files Added
- `ShitGPT.py` - Core model and training implementation
- `gui.py` - Graphical user interface
- `dataset.py` - Dataset loading and preprocessing utilities
- `live_training_plot.py` - Real-time plotting utilities
- `version.py` - Version information and changelog
- `setup.sh` - Linux/macOS setup script
- `setup.bat` - Windows setup script
- `setup.py` - Cross-platform Python setup script
- `requirements.txt` - Python dependencies
- `README.md` - Comprehensive documentation
- `LICENSE` - MIT License
- `CHANGELOG.md` - This file
- `.gitignore` - Git ignore rules

### Architecture Overview
```
DGS-GPT/
├── Core Model (ShitGPT.py)
│   ├── GPTLanguageModel
│   ├── Transformer Blocks
│   ├── Attention Mechanisms
│   └── MoE Layers
├── GUI Interface (gui.py)
│   ├── Text Generation Tab
│   ├── Training Tab
│   └── Optimization Tab
├── Training System
│   ├── Trainer Class
│   ├── Loss Monitoring
│   └── Progress Tracking
└── Optimization (Optuna)
    ├── Parameter Search
    ├── Trial Pruning
    └── Best Config Storage
```