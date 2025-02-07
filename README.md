# zbot-unit-tests

## Getting Started

1. Clone the repository

```bash
git clone git@github.com:kscalelabs/zbot-unit-tests.git
```

2. Make sure you're using Python 3.11 or greater

```bash
python --version  # Should show Python 3.11 or greater
```

3. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

4. Run the tests

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

### Roadmap

| Status | Name      | Description                 |
| ------ | --------- | --------------------------- |
| 🚧     | `test_00` | Inference speed test        |
| ✅     | `test_01` | Basic movement test         |
| 🚧     | `test_02` | Basic policy test           |
| ❌     | `test_03` | Inverse kinematics test     |
| ❌     | `test_04` | Motor system identification |

Key:

- ✅: Completed, other teams able to replicate and pass
- 🚧: Test is implemented, but issues persist
- ❌: Not implemented
