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

4. Start `kos-sim` server backend in a separate terminal:
```bash
kos-sim kbot-v1 --no-render # disable render of MuJoCo
```

5. Run the tests in another terminal:

Example:
```bash
python kbot_unit_tests/test_01.py
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

Please see issues: https://github.com/kscalelabs/kbot-unit-tests/issues


## Assets 
| Name | Description | Date Added | PR Link | Video Link |
| recording_00 | Isaac Gym stable policy with old urdf | 2025m02d15 | |  |
| recording_01 | Manual velocity scale | 2025m02d16 | https://photos.app.goo.gl/ZMFP185dwuCtf3ex5 |
