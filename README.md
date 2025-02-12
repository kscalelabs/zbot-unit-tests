# zbot-unit-tests

[Roadmap](https://github.com/orgs/kscalelabs/projects/20/views/1)

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
git clone git@github.com:kscalelabs/zbot-unit-tests.git
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
ks robots urdf pybullet zbot-v2 --fixed-base

# To check the MJCF model:
ks robots urdf mujoco zbot-v2
```

## Roadmap

| Status | Replicated | Name      | Description                 |
| ------ | ---------- | --------- | --------------------------- |
| 🚧     | ✅         | `test_00` | Inference speed test        |
| ✅     | ✅         | `test_01` | Basic movement test         |
| 🚧     | ❌         | `test_02` | Inverse kinematics test     |
| ❌     | ❌         | `test_03` | Motor system identification |
| 🚧     | ❌         | `test_04` | Basic policy test           |
| ✅     | ❌         | `test_05` | ZMP-based walking           |
| 🚧     | ❌         | `test_06` | IMU test                    |
