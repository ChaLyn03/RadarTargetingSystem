# Contributing to RadarTargetingSystem

Thank you for your interest in contributing to the RadarTargetingSystem project! This document provides guidelines and instructions for contributing.

## Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Help fellow contributors

## Getting Started

### 1. Fork and Clone
```bash
git clone https://github.com/YOUR_USERNAME/RadarTargetingSystem.git
cd RadarTargetingSystem
```

### 2. Set Up Development Environment
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
make install-dev
```

### 3. Create a Feature Branch
```bash
git checkout -b feature/your-feature-name
git checkout -b bugfix/your-bugfix-name
```

## Development Workflow

### Code Style
- Use `make format` to auto-format code with Black and isort
- Maximum line length: 120 characters
- Python 3.9+ syntax
- Type hints are encouraged but not required

### Before Committing
```bash
make lint       # Check code quality
make test       # Run tests
make format     # Format code
```

### Commit Messages
- Use clear, descriptive commit messages
- Reference issues using `#ISSUE_NUMBER`
- Example: `feat: add micro-Doppler simulation (#42)`

## Pull Request Process

1. **Keep PRs focused** — One feature or fix per PR
2. **Test your changes** — Run `make test` and verify functionality
3. **Update documentation** — Modify README.md or code comments if needed
4. **Write a clear description** — Explain what and why
5. **Request review** — Link to related issues

### PR Title Format
- `feat:` for new features
- `fix:` for bug fixes
- `docs:` for documentation updates
- `refactor:` for code refactoring
- `test:` for test additions/updates
- `chore:` for build, CI, or dependency updates

Example: `feat: add MUSIC direction finding algorithm`

## Code Organization

### Adding a New Detection Algorithm
1. Create `src/detection/your_method.py`
2. Implement: `def your_method(power_map: np.ndarray, **params) -> np.ndarray`
3. Add tests in `tests/detection/test_your_method.py`
4. Update `src/ui/pipeline.py` to include the new method
5. Document in README.md under "Detection Methods"

### Extending the Simulation
1. Modify `src/sim/fmcw.py` with new `SimulationConfig` fields
2. Update the `simulate_frame()` function logic
3. Add tests verifying new behavior
4. Update `.github/copilot-instructions.md` if introducing new patterns

### Training a Custom Classifier
1. Create dataset class in `src/models/dataset.py`
2. Extend `CLASSES` list if adding new target types
3. Update `src/models/training.py` to use new dataset
4. Test on synthetic patches

## Testing

### Running Tests
```bash
make test                    # Run all tests
pytest tests/sim/            # Run specific test module
pytest -v tests/             # Verbose output
pytest --cov src             # With coverage report
```

### Writing Tests
- Use `pytest` framework
- Place test files in `tests/` directory
- Name test files: `test_*.py`
- Name test functions: `test_*`

Example:
```python
from src.sim.fmcw import simulate_frame, SimulationConfig, Target

def test_simulate_frame_shape():
    config = SimulationConfig(n_chirps=16, n_samples=256)
    target = Target(range_m=100, velocity_mps=10)
    result = simulate_frame(config, [target])
    
    assert result.iq.shape == (16, 256)
    assert len(result.ranges_m) == 256
    assert len(result.doppler_hz) == 16
```

## Documentation

### Update README.md for:
- New algorithms or features
- Usage examples
- Parameters and configuration

### Update `.github/copilot-instructions.md` for:
- New architectural patterns
- Integration points
- Developer workflows

### Code Comments
- Use docstrings for functions and classes
- Explain complex physics formulas
- Reference equations: e.g., "Beat frequency: $f_b = 2KR/c$"

## Common Tasks

### Adding a Dependency
```bash
# Update requirements.txt
pip install your-package
pip freeze | grep your-package >> requirements.txt

# Or update pyproject.toml
# Then reinstall
pip install -e ".[dev]"
```

### Running the Dashboard Locally
```bash
make run
# Opens at http://localhost:8501
```

### Debugging
```python
# Add debug output to pipeline
output = run_pipeline(targets, cfg, device=device)
print(f"Power range: {output.power_map.min():.3f} to {output.power_map.max():.3f}")
print(f"Detections found: {len(output.detections)}")
```

## Performance Considerations

- FMCW simulation is CPU-bound; optimize the `simulate_frame()` loop for large chirp counts
- FFT operations dominate DSP processing; profiling with `cProfile` is recommended
- CNN inference benefits from GPU (CUDA); test both CPU and GPU paths
- Streamlit caches expensive computations; use `@st.cache_data` for non-interactive functions

## Reporting Issues

- Use GitHub Issues for bug reports
- Include reproducible steps
- Attach screenshots, logs, or minimal example code
- Label with appropriate tags (`bug`, `enhancement`, `documentation`)

## Questions?

Feel free to open a discussion or issue if you have questions. The project maintainers are happy to help!

---

**Thank you for contributing!** Your work makes RadarTargetingSystem better for everyone.
