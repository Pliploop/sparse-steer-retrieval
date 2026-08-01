# Configs Directory

This directory contains Hydra configuration files for training the Diff-GAR model.

## Structure

```
configs/
├── train.yaml              # Main training config (uses defaults)
├── test_config.yaml        # Minimal test config for validation
├── callbacks/              # Callback configurations
│   └── default.yaml
├── data/                   # Data module configurations
│   └── default.yaml
├── model/                  # Model configurations
│   └── default.yaml
├── trainer/                # Trainer configurations
│   └── default.yaml
├── paths/                  # Path configurations
│   └── default.yaml
├── extras/                 # Extra utilities configurations
│   └── default.yaml
└── logger/                 # Logger configurations (optional)
```

## Usage

### Basic Training

```bash
python diffgar/train.py
```

This will use the default configuration from `configs/train.yaml`.

### Test Config

To test that the refactor works correctly:

```bash
python diffgar/train.py --config-name=test_config
```

**Important**: Before running the test config, update the data paths in `test_config.yaml`:
- `data.datasets.test_musiccaps.root_dir`: Path to your audio data
- `data.datasets.test_musiccaps.new_dir`: Path to your extracted features

### Override Options

You can override any config value from the command line:

```bash
# Change batch size
python diffgar/train.py data.batch_size=64

# Change model
python diffgar/train.py model=custom_model

# Change number of epochs
python diffgar/train.py trainer.max_epochs=100

# Combine multiple overrides
python diffgar/train.py data.batch_size=64 trainer.max_epochs=50 seed=123
```

### Using Different Configs

```bash
# Use a specific data config
python diffgar/train.py data=custom_data

# Use a specific model config  
python diffgar/train.py model=custom_model

# Use a specific trainer config
python diffgar/train.py trainer=custom_trainer
```

## Configuration Hierarchy

The configuration system uses Hydra's composition pattern:

1. **Default configs** are loaded from the `defaults` list in `train.yaml`
2. **Specific configs** can be specified via command line (e.g., `data=musiccaps`)
3. **Overrides** can be applied directly (e.g., `data.batch_size=64`)

Priority (highest to lowest):
1. Command-line overrides
2. Specific config files
3. Default config files
4. Base `train.yaml` values

## Creating New Configs

### New Data Config

Create `configs/data/my_dataset.yaml`:

```yaml
# @package _global_

data:
  _target_: diffgar.dataloading.dataloaders.TextAudioDataModule
  datasets:
    my_dataset:
      _target_: diffgar.dataloading.datasets.TextAudioDataset
      # ... your config
  batch_size: 32
  num_workers: 4
```

Then use: `python diffgar/train.py data=my_dataset`

### New Model Config

Create `configs/model/my_model.yaml`:

```yaml
# @package _global_

model:
  _target_: diffgar.models.ldm.diffusion.LightningDiffGar
  # ... your config
```

Then use: `python diffgar/train.py model=my_model`

## Integration with config/train_ldm

The old `config/train_ldm/` structure can still be used by creating appropriate config files in this directory that reference those configurations. The new structure provides better modularity and composability.

