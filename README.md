# Cross-Species Vocalization Classifier

This repository contains code to train, evaluate and run a cross-species vocalization classifier built on top of a fine-tuned Whisper encoder with lightweight temporal and classification heads.

This README is ordered from data acquisition → training → evaluation → inference/demo. Follow the sections in order for a reproducible workflow.

**Prerequisites**
- **Python**: 3.8 or newer
- **CUDA** (optional): required for GPU training. The code will run on CPU if a GPU is not available.

**Install (recommended)**
- Create and activate a virtual environment:

	- Windows (PowerShell):
		```powershell
		python -m venv .venv
		.\.venv\Scripts\Activate.ps1
		```

	- Unix / macOS:
		```bash
		python -m venv .venv
		source .venv/bin/activate
		```

- Install dependencies:

	```bash
	pip install -r requirements.txt
	```

If you cannot or do not want to use the bundled `requirements.txt`, install the main dependencies manually: `torch`, `torchaudio`, `numpy`, `pandas`, `librosa`, `soundfile`, `openai-whisper`, `streamlit`, `plotly`, `requests`, `tqdm`.

**1) Download the data**

The repository expects raw data under `data/raw/`.

- ESC-50 (small labeled environmental sound dataset)
	- Run:

		```bash
		python data/downloads/download_esc50.py
		```

	- The script downloads and extracts the ESC-50 repository to `data/raw/ESC-50` and will create an `audio/` and `meta/esc50.csv` structure that the dataset loader expects.

- Freesound (optional, for additional animal/machinery examples)
	- The provided script `data/downloads/download_freesound.py` uses the Freesound Python client and requires an API key.
	- Edit `data/downloads/download_freesound.py` and set `API_KEY` to your Freesound API key, or modify the script to read an environment variable before running.
	- Run:

		```bash
		python data/downloads/download_freesound.py
		```

Notes:
- The dataset loader (`training/dataset.py`) will search for ESC-50 at either `data/raw/ESC-50-master/audio` or `data/raw/ESC-50/ESC-50-master/audio`. If extraction produced a different layout, move files so the loader can find `esc50.csv` and the `audio/` folder.
- Freesound downloads are saved to `data/raw/freesound/<query_name>`.

**2) Prepare / inspect the data**
- No separate processing script is strictly required. `training.dataset.CrossSpeciesDataset` reads from `data/raw` and infers labels from `esc50.csv` (ESC-50) and from folder names for Freesound data.
- If you prefer a pre-processed set, place mel spectrogram arrays or reorganized files under `data/processed/` and adapt the loader accordingly.

**3) Quick inference using provided checkpoints**
- The `checkpoints/` folder may contain saved models (for example `checkpoints/final_model.pt`). To classify a single file:

	```bash
	python inference/classifier.py path/to/audio.wav --model checkpoints/final_model.pt
	```

- Batch process a folder and save results to CSV:

	```bash
	python inference/batch_process.py path/to/folder --output results.csv --workers 4
	```

- Run live microphone classification (records then classifies):

	```bash
	python inference/live_mic.py --duration 5
	```

**4) Training (fine-tuning)**

I added a small helper script `run_training.py` to run the three fine-tuning phases described by the project. The high-level flow is:

- Phase 1: Train new heads with the Whisper encoder frozen.
- Phase 2: Unfreeze top encoder layers and continue training.
- Phase 3: Train domain-specific adapters (if enabled).

Use the helper script (defaults are defined in `models/config.py`):

```bash
python run_training.py --phase 1            # run phase 1 only
python run_training.py --phase 2            # run phase 2 only
python run_training.py --phase 3            # run phase 3 only
python run_training.py --phase all          # run phases 1 → 2 → 3 sequentially
```

The helper calls into `training/fine_tune.FineTuner` and writes logs under `logs/` and checkpoints under `checkpoints/` (see `models/config.py` for defaults).

If you prefer to integrate the training loop into your own scripts, instantiate `FineTuner` directly:

```python
from models.config import get_configs
from models.cross_species_model import CrossSpeciesVocalizationModel
from training.fine_tune import FineTuner

mcfg, tcfg = get_configs(whisper_size='tiny')
model = CrossSpeciesVocalizationModel(mcfg)
tuner = FineTuner(model, mcfg, tcfg)
tuner.phase1()
```

**5) Full evaluation**
- The evaluation utilities are in `evaluation/`. To run the full test evaluation that computes per-class metrics, top-k scores and inference speed, run:

```bash
python evaluation/evaluate.py
```

Outputs and summaries will be printed and can be saved by modifying the evaluation runner (the script includes a `save_results` helper).

**6) Streamlit demo**
- There is a Streamlit app in `demo/app.py` for interactive classification and batch processing.

	```bash
	streamlit run demo/app.py
	```

This app loads the classifier from `checkpoints/final_model.pt` by default. If you do not have a checkpoint, the demo will warn and run with an untrained model.

**Files of interest**
- **Model definition**: `models/cross_species_model.py`
- **Configurations**: `models/config.py`
- **Training pipeline**: `training/fine_tune.py`
- **Dataset loader**: `training/dataset.py`
- **Inference utilities**: `inference/classifier.py`, `inference/batch_process.py`, `inference/live_mic.py`
- **Data download helpers**: `data/downloads/download_esc50.py`, `data/downloads/download_freesound.py`

**Troubleshooting & notes**
- On Windows, `training/dataset.py` sets `TORCHAUDIO_BACKEND=soundfile` to avoid some `torchaudio` build issues; if you see import errors for `torchaudio`, try installing a compatible wheel for your platform or rely on `librosa` fallback.
- The Freesound script requires a valid API key; the provided script contains a placeholder—replace it before running.
- Whisper model weights will be downloaded the first time `whisper.load_model(...)` is called; ensure you have internet connectivity and enough disk space.

**License & citation**
- If you reuse the model or dataset subsets, please cite ESC-50 and OpenAI Whisper as appropriate.
 
