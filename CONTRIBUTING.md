# Contributing to SpikeLink

Thank you for your interest in contributing to **SpikeLink**.

SpikeLink is a spike-native transport protocol intended for long-lived, reproducible neuromorphic research and infrastructure. Contributions are welcome, but they are guided by a clear scope and discipline to preserve correctness, stability, and interpretability.

---

## Development Setup

```bash
# Clone the repository
git clone https://github.com/lightborneintelligence/spikelink.git
cd spikelink

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install in development mode
pip install -e ".[dev,full]"

# Run tests
pytest
````

---

## Code Style and Tooling

SpikeLink follows strict, predictable tooling to ensure long-term maintainability.

We use:

* **black** — code formatting
* **ruff** — linting and static checks
* **mypy** — type checking

Before submitting any changes, please run:

```bash
black src tests
ruff check src tests
mypy src
```

Pull requests that do not pass these checks will not be merged.

---

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=spikelink --cov-report=html

# Run a specific test file
pytest tests/unit/test_codec.py
```

All new functionality **must** include tests.

---

## Contribution Scope

### Appropriate Contributions

We welcome contributions that improve or extend:

* Code correctness and robustness
* Documentation clarity
* Test coverage
* EBRAINS ecosystem compatibility (Neo, Elephant, PyNN)
* Performance *within documented bounds*
* Bug fixes and reproducibility improvements

### Out-of-Scope Contributions

The following are intentionally **out of scope** for public contribution:

* Changes that alter protocol semantics
* Unbounded performance claims or benchmarks
* Internal calibration logic
* Long-run endurance or adversarial stress pipelines
* Cryptographic or protected validation artifacts

These areas are governed under controlled review (see `SECURITY.md`).

---

## Submitting Changes

1. Fork the repository
2. Create a feature branch

   ```bash
   git checkout -b feature/your-feature-name
   ```
3. Make your changes
4. Add or update tests as required
5. Ensure all tests and checks pass
6. Submit a pull request with a clear description

---

## Pull Request Guidelines

To keep SpikeLink stable and trustworthy:

* Keep PRs **focused** on a single change
* Avoid refactors unrelated to the stated goal
* Update documentation where behavior changes
* Use clear, descriptive commit messages
* Do not introduce breaking changes without discussion

Large or architectural changes should be proposed via a GitHub discussion **before** implementation.

---

## Reporting Issues

When reporting issues, please include:

* Python version
* SpikeLink version
* Operating system
* Minimal reproducible example
* Expected vs actual behavior

Incomplete reports may be closed for clarification.

---

## Questions and Discussion

For general questions or design discussions:

* Open a GitHub Discussion
* Or contact: **[contact@lightborneintelligence.com](mailto:contact@lightborneintelligence.com)**

For security-related matters, see `SECURITY.md`.

---

**Lightborne Intelligence**
*Truth > Consensus · Sovereignty > Control · Coherence > Speed*
