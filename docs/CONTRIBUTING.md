# Contributing to DGS-GPT

Thank you for your interest in contributing to DGS-GPT! This document provides guidelines for contributing to the project.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/your-username/DGS-GPT.git
   cd DGS-GPT
   ```
3. **Set up the development environment**:
   ```bash
   # Use one of the setup scripts
   python setup.py  # Cross-platform
   # OR
   ./setup.sh       # Linux/macOS
   # OR
   setup.bat        # Windows
   ```

## Development Guidelines

### Code Style
- Follow PEP 8 for Python code style
- Use meaningful variable and function names
- Add docstrings to classes and functions
- Keep functions focused and concise
- Use type hints where appropriate

### Testing
- Test your changes thoroughly before submitting
- Ensure the GUI loads and functions correctly
- Test both CUDA and CPU execution paths
- Verify hyperparameter optimization works

### Documentation
- Update the README.md if you add new features
- Add entries to CHANGELOG.md for significant changes
- Document any new configuration options
- Include usage examples for new features

## Types of Contributions

### Bug Reports
- Use the GitHub issue tracker
- Include detailed reproduction steps
- Provide system information (OS, Python version, GPU)
- Include error messages and stack traces

### Feature Requests
- Describe the feature and its use case
- Explain why it would be valuable
- Consider implementation complexity

### Code Contributions
- Start with small, focused changes
- One feature or fix per pull request
- Include tests for new functionality
- Update documentation as needed

## Pull Request Process

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**:
   - Write clean, well-documented code
   - Follow the existing code style
   - Test thoroughly

3. **Commit your changes**:
   ```bash
   git add .
   git commit -m "Add feature: description of your changes"
   ```

4. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

5. **Create a Pull Request**:
   - Use a clear, descriptive title
   - Explain what your changes do
   - Reference any related issues
   - Include screenshots for UI changes

## Development Areas

### High Priority
- **Tokenization**: Add subword tokenization (BPE, SentencePiece)
- **Multi-GPU**: Implement distributed training support
- **Sampling**: Advanced text generation strategies (nucleus, top-k, temperature)
- **Performance**: Optimization and memory usage improvements

### Medium Priority
- **Model Architectures**: Additional transformer variants
- **Data Loading**: More efficient dataset handling
- **Evaluation**: Automatic model evaluation metrics
- **Export**: Model export to ONNX or other formats

### Low Priority
- **UI Improvements**: Additional GUI features and themes
- **Documentation**: More examples and tutorials
- **Integration**: API for external applications

## Code Review Criteria

Pull requests will be evaluated based on:
- **Functionality**: Does it work as intended?
- **Code Quality**: Is it clean, readable, and maintainable?
- **Performance**: Does it impact performance negatively?
- **Compatibility**: Does it work across platforms?
- **Documentation**: Is it properly documented?

## Getting Help

- **GitHub Issues**: For bug reports and feature requests
- **Discussions**: For questions and general discussion
- **Code Review**: Ask for feedback on draft pull requests

## License

By contributing to DGS-GPT, you agree that your contributions will be licensed under the MIT License.

## Recognition

Contributors will be acknowledged in the project documentation and release notes.

Thank you for helping make DGS-GPT better! 🚀
