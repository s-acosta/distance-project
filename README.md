# Distance Project

Code for all experiments related to the distance coding project.

## Project Structure

```
distance-project/
├── radial-analysis/        Radial stimulus experiment (main analysis framework)
│   ├── radial-notebooks/   Analysis notebooks (decoding, single-cell, multi-session, etc.)
│   ├── src/                Python analysis pipeline (data loading, analysis, visualization, decoding)
│   └── run-scripts/        Batch processing scripts
│
├── linear-analysis/        Linear stimulus experiment
│   └── linear-notebooks/   Analysis notebooks (binocular, time params, spike params, etc.)
│
├── disparity-analysis/     Disparity/mouse goggles experiment
│   └── disparity-notebooks/  Analysis notebooks and results
│
├── visual-analysis/        Visual tuning and sign mapping
│   ├── visual-notebooks/   Orientation topology and tuning notebooks
│   ├── visual-stimuli/     Stimulus generation scripts and frames
│   └── sign-maps/          MATLAB sign map computation
│
├── hardware-control/       Shared hardware control code
│   ├── calibration/        Camera and distance calibration (MATLAB + notebooks)
│   ├── radial-device/      Radial device firmware (PlatformIO/C++)
│   ├── linear-device/      Linear device control (Arduino + MATLAB)
│   ├── disparity-device/   Disparity stimulus delivery (Teensy + MATLAB)
│   ├── camera-acquisition/ Camera triggering and acquisition
│   └── misc/               Legacy hardware code (radial-device v0, etc.)
│
├── legacy/                 Archived older analysis code and notebooks
├── path_setup.py           Import helper for notebooks and scripts
└── .gitignore
```

## Setup

Notebooks and scripts use `sys.path.append` to add the project root, then import from `src.*`:

```python
import sys
sys.path.append(r"C:\Users\acostamendoza\projects\distance-project")
from src.data.RadialDevice import RadialDevice
```
