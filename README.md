# zbot-unit-tests

## Getting Started

1. Clone the repository

```bash
git clone git@github.com:kscalelabs/zbot-unit-tests.git
```

2. Install dependencies

```bash
pip install .
```

3. Run the tests

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
