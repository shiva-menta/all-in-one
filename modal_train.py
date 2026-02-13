"""
Modal training script for All-In-One Music Structure Analyzer.

This script allows you to train the model on Modal's cloud infrastructure with GPU support.

Usage:
    # First, set up Modal (one-time):
    pip install modal
    modal setup

    # Set your W&B API key as a Modal secret:
    modal secret create wandb WANDB_API_KEY=<your-key>

    # Run training for a single fold:
    modal run modal_train.py --fold 0

    # Run training for all folds in parallel:
    modal run modal_train.py --all-folds

    # Run preprocessing (required before first training):
    modal run modal_train.py --preprocess
"""

import modal

# Define the Modal app
app = modal.App("allin1-training")

# Create a volume for persistent data storage (dataset, checkpoints)
volume = modal.Volume.from_name("allin1-data", create_if_missing=True)

# Define the container image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "git",
        "ffmpeg",
        "libsndfile1",
    )
    .pip_install(
        # Core dependencies
        "numpy>=1.24",
        "torch>=2.0.0",
        "torchaudio>=2.0.0",
        # Install natten from source for CUDA support
        "natten>=0.20.0",
        # Audio processing
        "librosa>=0.10.0",
        "demucs>=4.0.0",
        "madmom",
        # ML framework
        "lightning>=2.0.0",
        "timm>=0.9.0",
        # Config & logging
        "hydra-core>=1.3.0",
        "omegaconf>=2.3.0",
        "wandb>=0.16.0",
        # Evaluation
        "mir_eval>=0.8.0",
        # Misc
        "huggingface_hub>=0.20.0",
        "matplotlib>=3.7.0",
        "scikit-learn",
        "tqdm",
    )
    # Install the allin1 package in editable mode
    .pip_install("allin1[train]")
)

# Mount the data volume at /data
DATA_PATH = "/data/harmonix"
VOLUME_MOUNT = "/data"


@app.function(
    image=image,
    gpu="A100",  # Use A100 GPU for training (can also use "T4", "A10G", "H100")
    timeout=60 * 60 * 24,  # 24 hour timeout for long training runs
    volumes={VOLUME_MOUNT: volume},
    secrets=[modal.Secret.from_name("wandb")],
)
def train(fold: int = 0, **kwargs):
    """Train the model for a specific fold."""
    import os
    import subprocess

    # Set up W&B
    os.environ["WANDB_DIR"] = "/data/wandb"
    os.makedirs(os.environ["WANDB_DIR"], exist_ok=True)

    # Build the command
    cmd = [
        "allin1-train",
        f"fold={fold}",
        f"data.path_base_dir={DATA_PATH}/",
        f"data.path_track_dir={DATA_PATH}/tracks/",
        f"data.path_demix_dir={DATA_PATH}/demix/",
        f"data.path_feature_dir={DATA_PATH}/features/",
        f"data.path_no_demixed_feature_dir={DATA_PATH}/features_no_demixed/",
        f"data.path_metadata={DATA_PATH}/metadata.csv",
    ]

    # Add any additional kwargs
    for key, value in kwargs.items():
        cmd.append(f"{key}={value}")

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=True)

    # Commit volume changes to persist checkpoints
    volume.commit()

    return f"Training completed for fold {fold}"


@app.function(
    image=image,
    gpu="A100",
    timeout=60 * 60 * 12,  # 12 hour timeout for preprocessing
    volumes={VOLUME_MOUNT: volume},
)
def preprocess():
    """Preprocess the dataset (source separation and feature extraction)."""
    import os
    import subprocess

    # Build the command
    cmd = [
        "allin1-preprocess",
        f"data.path_base_dir={DATA_PATH}/",
        f"data.path_track_dir={DATA_PATH}/tracks/",
        f"data.path_demix_dir={DATA_PATH}/demix/",
        f"data.path_feature_dir={DATA_PATH}/features/",
    ]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=True)

    # Commit volume changes to persist preprocessed features
    volume.commit()

    return "Preprocessing completed"


@app.function(
    image=image,
    volumes={VOLUME_MOUNT: volume},
)
def upload_data(local_path: str):
    """
    Upload local data to the Modal volume.

    Usage:
        modal run modal_train.py::upload_data --local-path ./data/harmonix
    """
    import shutil
    import os

    # Copy files to volume
    if os.path.exists(DATA_PATH):
        shutil.rmtree(DATA_PATH)
    shutil.copytree(local_path, DATA_PATH)

    volume.commit()
    return f"Uploaded data from {local_path} to {DATA_PATH}"


@app.function(
    image=image,
    volumes={VOLUME_MOUNT: volume},
)
def list_data():
    """List the contents of the data volume."""
    import os

    result = []
    for root, dirs, files in os.walk(VOLUME_MOUNT):
        level = root.replace(VOLUME_MOUNT, '').count(os.sep)
        indent = ' ' * 2 * level
        result.append(f'{indent}{os.path.basename(root)}/')
        subindent = ' ' * 2 * (level + 1)
        for file in files[:10]:  # Limit to first 10 files per directory
            result.append(f'{subindent}{file}')
        if len(files) > 10:
            result.append(f'{subindent}... and {len(files) - 10} more files')

    return '\n'.join(result)


@app.local_entrypoint()
def main(
    fold: int = 0,
    all_folds: bool = False,
    preprocess_data: bool = False,
    list_files: bool = False,
):
    """
    Main entrypoint for Modal training.

    Args:
        fold: Which fold to train (0-7)
        all_folds: Train all 8 folds in parallel
        preprocess_data: Run preprocessing instead of training
        list_files: List files in the data volume
    """
    if list_files:
        result = list_data.remote()
        print(result)
        return

    if preprocess_data:
        print("Starting preprocessing...")
        result = preprocess.remote()
        print(result)
        return

    if all_folds:
        print("Starting training for all 8 folds in parallel...")
        # Use starmap to run all folds in parallel
        results = list(train.starmap([(i,) for i in range(8)]))
        for i, result in enumerate(results):
            print(f"Fold {i}: {result}")
    else:
        print(f"Starting training for fold {fold}...")
        result = train.remote(fold=fold)
        print(result)
