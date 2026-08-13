# MANO and SMPL+H

MANO is a hand model, while SMPL+H models the human body and hands.

## License and Resources

- [MANO website](http://mano.is.tue.mpg.de)
- [MANO/SMPL+H paper](http://files.is.tue.mpg.de/dtzionas/MANO/paper/Embodied_Hands_SiggraphAsia2017.pdf)
- [Model and data downloads](http://mano.is.tue.mpg.de/downloads)

The downloads page provides scans, alignments, model files, and Python code
for MANO and SMPL+H.

For comments or questions, contact [mano@tue.mpg.de](mailto:mano@tue.mpg.de).

## System Requirements

Supported operating systems:

- macOS
- Linux

Python dependencies:

- [NumPy and SciPy](https://www.scipy.org/)
- [Chumpy](https://github.com/mattloper/chumpy)
- [OpenCV](https://opencv.org/)

## Getting Started

### 1. Extract the code

Extract `mano.zip` into your home directory or another suitable location.

### 2. Configure `PYTHONPATH`

Add the following lines to `~/.bash_profile` on macOS or `~/.bashrc` on Linux.
Replace `~/mano` if you extracted the code elsewhere.

```bash
MANO_LOCATION=~/mano
export PYTHONPATH="$PYTHONPATH:$MANO_LOCATION"
```

Open a new terminal and verify the configuration:

```bash
echo "$PYTHONPATH"
```

### 3. Run the examples

Navigate to the `mano/webuser/hello_world` directory and run one of these
commands:

```bash
python MANO___hello_world.py
python MANO___render.py
python SMPL+H___hello_world.py
python SMPL+H___render.py
```

Each example requires the Python dependencies listed above.

## Acknowledgements

The code is based on the [SMPL release code](http://smpl.is.tue.mpg.de).
Thanks to Matthew Loper and Naureen Mahmood.
