# K-Bot Unit Tests

## Getting Started

1. Set up Git LFS and pull large files

```bash
# Install Git LFS
git lfs install

# Pull large files (URDF models, neural networks, etc.)
git lfs pull
```

2. Clone the repository

```bash
git clone git@github.com:kscalelabs/kbot-unit-tests.git
```

3. Make sure you're using Python 3.11 or greater

```bash
python --version  # Should show Python 3.11 or greater
```

4. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

5. Run the tests

```bash
make test
```

### Additional Tests

Check that the URDF and MJCF models are realistic:

```bash
# To check the URDF model:
ks robots urdf pybullet kbot-v1 --fixed-base

# To check the MJCF model:
ks robots urdf mujoco kbot-v1
```

## Roadmap

| Status | Replicated | Name      | Description                                |
| ------ | ---------- | --------- | ------------------------------------------ |
| 🚧     | ❌         | `test_00` | kos-sim matching real robot test          |
| 🚧     | ❌         | `test_01` | play recorded actions test                |
