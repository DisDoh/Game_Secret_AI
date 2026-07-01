# SecreatAI

SecreatAI is a game prototype built around AI-encoded file transfer. The goal is to train an autoencoder, use that model to transform a file into a `.aiz` file protected by a mask password, then challenge another player to reconstruct the file with their own model or their own analysis methods.

> Important: this project is a playful and educational experiment about autoencoders, binary representations, and pattern analysis. It is not a reliable encryption tool for protecting sensitive data.

## Game Objective

Each player has an autoencoder model trained separately. One player encodes a file with their model, shares only the `.aiz` file, and the opponent tries to decode it.

The encoding player scores points if the opponent cannot reconstruct the file. The opponent scores points if they manage to produce a correct reconstruction.

## Features

- Tkinter GUI for training, encoding, and decoding.
- Train or resume training of a `.pkl` model.
- Encode files into `.aiz` with a random salt and mask password.
- Decode files into the `decoded/` folder.
- Training loss curve in the interface.
- Pattern analysis tool for comparing two `.aiz` files.
- Separate scripts for training, encoding, and decoding.

## Requirements

- Python 3.10, configured by `.python-version`.
- `pip`.
- `tkinter` for the graphical interface.

On some Linux distributions, `tkinter` must be installed through the system package manager, for example:

```bash
sudo apt install python3-tk
```

## Installation

From the project folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export MPLCONFIGDIR=.matplotlib-cache
```

The current Python dependencies are intentionally lightweight:

- `numpy`
- `matplotlib`

The project also uses modules from the Python standard library, including `pickle`, `lzma`, `hashlib`, `tarfile`, `threading`, and `tkinter`.

## Quick Start

The graphical interface is the easiest way to use the project:

```bash
python SecreatAI_GUI.py
```

In the interface:

1. Choose an existing model or click `New Model`.
2. Click `Train / Resume` to create or continue training the model.
3. Choose a file with `Browse`.
4. Click `Encode`, enter a password, then produce a `.aiz` file.
5. Click `Decode`, enter the same password, then reconstruct a file from a `.aiz`.

Decoded files are written to `decoded/`. In the interface, files chosen outside the project folder are first copied into the application folder before encoding or decoding.

## Script Usage

The main scripts can also be called from Python.

### Train a Model

```bash
python Training_only.py
```

By default, this script creates or resumes `model.pkl`.

To use another model name:

```bash
python -c "import Training_only; Training_only.model_name='alice.pkl'; Training_only.main()"
```

### Encode a File

```bash
python -c "import Encode_only; Encode_only.main('model.pkl', 'mon_fichier.pdf', mask_password='secret')"
```

If the internal reconstruction reaches 100%, a `mon_fichier.pdf.aiz` file is created in the same folder as the source file, unless an output folder is provided.

Example with an output folder:

```bash
python -c "import Encode_only; Encode_only.main('model.pkl', 'mon_fichier.pdf', output_dir='encoded', mask_password='secret')"
```

If `mask_password` is not provided, the script asks for the password in the terminal.

### Decode a `.aiz` File

```bash
python -c "import Decode_only; Decode_only.main('model.pkl', 'mon_fichier.pdf.aiz', mask_password='secret')"
```

By default, the result is written to `decoded/` with the `.aiz` suffix removed. For `.aiz` files produced with a password, the same password is required for decoding.

### Compare Two `.aiz` Files

```bash
python pattern_aiz_compare.py fichier1.aiz fichier2.aiz
```

This script displays statistics about 64-bit blocks, repetitions, entropy, and Hamming distance. It helps identify whether visible patterns remain in encoded files.

## Proposed Rules

1. Each player trains their own `.pkl` model.
2. The active player chooses a source file.
3. They encode the file into `.aiz` and verify that their own model can reconstruct the file.
4. They keep their password secret or define a sharing rule before the round.
5. They send only the `.aiz` file to the opponent.
6. The opponent tries to decode the file with their model or with any other analysis strategy.
7. The original file is revealed.
8. Points are awarded:
   - opponent wins if the file is correctly reconstructed;
   - encoder wins if the file remains undecipherable.
9. Roles rotate in the next round.

## File Structure

```text
.
|-- SecreatAI_GUI.py                 # Graphical interface
|-- Training_only.py                 # Train/resume a model
|-- Encode_only.py                   # Encode a file to .aiz
|-- Decode_only.py                   # Decode a .aiz
|-- Autoencoder_Encoder_Decoder.py   # Historical combined version
|-- Autoencoder_Encoder_Decoder_timed.py
|-- pattern_aiz_compare.py           # Pattern analysis between .aiz files
|-- Compare_files.py                 # Simple binary file comparison
|-- requirements.txt
|-- model.pkl                        # Default model, if present
`-- decoded/                         # Decoded outputs
```

## Generated Files

- `*.pkl`: trained models. The provided `model.pkl` file is used as the default model.
- `*.aiz`: encoded files. These outputs are generated locally and ignored by Git.
- `decoded/`: reconstructed files.
- `.matplotlib-cache/`: local Matplotlib cache.
- `__pycache__/`: Python cache.

These files are ignored by `.gitignore` when they are temporary or generated.

## Technical Notes

The model works on 8-bit input blocks and produces 64-bit encoded representations. The encoded data is then compressed with `lzma`, then masked with a random salt before the `.aiz` file is written. If a password is provided, it is combined with the salt using `sha256` to produce the mask stream.

Training uses a custom dense neural network implementation and the Adam optimizer with `numpy`. Training and validation losses are stored in the `.pkl` file, allowing the interface to display the history.

## Known Limitations

- The `.aiz` format is not cryptographic encryption.
- The mask password makes analysis harder, but it does not replace real encryption.
- The same model can reveal exploitable regularities depending on the files and training level.
- The CLI scripts do not have an `argparse` interface yet; advanced calls are therefore made through `python -c` or through the GUI.
- `pickle` must not be used with untrusted models: loading a `.pkl` file from an unknown source can execute arbitrary code.

## Troubleshooting

### Matplotlib-Related Error

Define a local cache folder:

```bash
export MPLCONFIGDIR=.matplotlib-cache
```

### `No module named tkinter` Error

Install the Tkinter system package matching your Python. On Debian/Ubuntu:

```bash
sudo apt install python3-tk
```

### Decoding Produces an Incorrect File

Check that the `.aiz` was decoded with the same model and the same password used for encoding. In the game, using another model is part of the challenge, but exact reconstruction is then not guaranteed.

### `Invalid password or corrupted AIZ payload` Error

The provided password does not match the `.aiz` file, or the encoded file is incomplete/corrupted. Try again with the password used during encoding.

## License

This project is distributed under the MIT license. See [LICENSE](LICENSE).
