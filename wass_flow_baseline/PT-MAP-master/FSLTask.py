import os
import pickle
import numpy as np
import torch

# ========================================================
#   Path helpers
# ========================================================

# Non-medical datasets (keyed by logical name -> pickle from save_plk.py)
_DATASET_FEATURE_FILES = {
    "miniimagenet": "./checkpoints/miniImagenet/WideResNet28_10_S2M2_R/last/output.plk",
    "cub":          "./checkpoints/CUB/WideResNet28_10_S2M2_R/last/output.plk",
    "cifar":        "./checkpoints/cifar/WideResNet28_10_S2M2_R/last/output.plk",
    "cross":        "./checkpoints/cross/WideResNet28_10_S2M2_R/last/output.plk",
}

# Medical subsets: feature path is constructed at runtime from
#   ./checkpoints/medical/<subset>/<model>_<method>/last/output.plk
# The actual model and method used are read from command-line flags
# passed to test_standard.py and stored in the module-level variable
# below (set via FSLTask.configure_medical()).

_cacheDir = "./cache"
_maxRuns = 10000

# ========================================================
#   Module-internal state
# ========================================================

_randStates = None
_rsCfg = None

# Medical feature path, set by configure_medical() called from test_standard.py
_medical_feature_path = None


# ========================================================
#   Internal helpers
# ========================================================

def _load_pickle(file):
    with open(file, 'rb') as f:
        data = pickle.load(f)
    labels = [np.full(shape=len(data[key]), fill_value=key)
              for key in data]
    flat = [feat for key in data for feat in data[key]]
    return {
        'data': torch.FloatTensor(np.stack(flat, axis=0)),
        'labels': torch.LongTensor(np.concatenate(labels)),
    }


# ========================================================
#   Public API
# ========================================================

data = None
labels = None
dsName = None


def configure_medical(subset, model='WideResNet28_10', method='S2M2_R'):
    """Call this from test_standard.py before loadDataSet() for medical subsets.

    Args:
        subset: the medical subset name, e.g. '胃_1query'
        model: backbone name, e.g. 'WideResNet28_10' or 'ResNet18'
        method: training method, e.g. 'S2M2_R' or 'rotation'
    """
    global _medical_feature_path
    _medical_feature_path = (
        f"./checkpoints/medical/{subset}/{model}_{method}/last/output.plk"
    )


def get_feature_path(dsname):
    """Return the feature pickle path for any dataset, including medical."""
    if dsname in _DATASET_FEATURE_FILES:
        return _DATASET_FEATURE_FILES[dsname]
    if _medical_feature_path is not None:
        return _medical_feature_path
    # Auto-detect: try ResNet18 first, then WideResNet28_10
    for model in ['ResNet18', 'WideResNet28_10']:
        for method in ['S2M2_R', 'rotation']:
            candidate = f"./checkpoints/medical/{dsname}/{model}_{method}/last/output.plk"
            if os.path.isfile(candidate):
                return candidate
    return None


def loadDataSet(dsname):
    global dsName, data, labels, _randStates, _rsCfg, data

    feature_path = get_feature_path(dsname)
    if feature_path is None or not os.path.isfile(feature_path):
        known = ', '.join(sorted(_DATASET_FEATURE_FILES))
        raise NameError(
            f"Dataset '{dsname}' not recognised and no feature file found at\n"
            f"  {feature_path or './checkpoints/medical/<subset>/<model>_<method>/last/output.plk'}\n"
            f"Known datasets: {known}\n"
            f"For a medical subset, call FSLTask.configure_medical('<subset>', '<model>', '<method>') "
            f"before loadDataSet(), or ensure the pickle exists at the path above."
        )

    dsName = dsname
    _randStates = None
    _rsCfg = None

    print(f'[FSLTask] loading features from: {feature_path}')
    dataset = _load_pickle(feature_path)

    # Count items per class
    unique, counts = torch.unique(dataset['labels'], return_counts=True)
    _min_examples = int(counts.min().item())
    print(f"[FSLTask] guaranteed items per class: {_min_examples}")

    # Build tensor [n_classes, _min_examples, n_features]
    n_classes = int(unique.numel())
    n_feat = int(dataset['data'].shape[1])
    data = torch.zeros((n_classes, _min_examples, n_feat))
    remaining = dataset['labels'].clone()
    ci = 0
    while remaining.numel() > 0:
        target = remaining[0]
        idxs = torch.where(dataset['labels'] == target)[0][: _min_examples]
        data[ci] = dataset['data'][idxs]
        remaining = remaining[remaining != target]
        ci += 1
    labels = dataset['labels'].clone()
    print(f"[FSLTask] {n_classes} classes, {_min_examples} items/class, "
          f"feature dim {n_feat}")


def GenerateRun(iRun, cfg, regenRState=False, generate=True):
    global _randStates
    if not regenRState and _randStates is not None:
        np.random.set_state(_randStates[iRun])

    classes = np.random.permutation(np.arange(data.shape[0]))[:cfg["ways"]]
    shuffle_idx = np.arange(data.shape[1])
    dataset = None
    if generate:
        dataset = torch.zeros((cfg['ways'], cfg['shot'] + cfg['queries'], data.shape[2]))
    for i in range(cfg['ways']):
        shuffle_idx = np.random.permutation(shuffle_idx)
        if generate:
            dataset[i] = data[classes[i], shuffle_idx, :][:cfg['shot'] + cfg['queries']]
    return dataset


def ClassesInRun(iRun, cfg):
    global _randStates
    if _randStates is not None:
        np.random.set_state(_randStates[iRun])
    classes = np.random.permutation(np.arange(data.shape[0]))[:cfg["ways"]]
    return classes


def setRandomStates(cfg):
    global _randStates, _maxRuns, _rsCfg
    if _rsCfg == cfg:
        return

    rsFile = os.path.join(_cacheDir, "RandStates_{}_s{}_q{}_w{}".format(
        dsName, cfg['shot'], cfg['queries'], cfg['ways']))
    if not os.path.exists(rsFile):
        print(f"{rsFile} not found, regenerating...")
        np.random.seed(0)
        _randStates = []
        for iRun in range(_maxRuns):
            _randStates.append(np.random.get_state())
            GenerateRun(iRun, cfg, regenRState=True, generate=False)
        with open(rsFile, 'wb') as f:
            pickle.dump(_randStates, f)
    else:
        print("reloading random states from file....")
        try:
            with open(rsFile, 'rb') as f:
                _randStates = pickle.load(f)
        except Exception:
            _randStates = torch.load(rsFile, weights_only=False)
    _rsCfg = cfg


def GenerateRunSet(start=None, end=None, cfg=None):
    global _maxRuns
    if start is None:
        start = 0
    if end is None:
        end = _maxRuns
    if cfg is None:
        cfg = {"shot": 1, "ways": 5, "queries": 15}

    setRandomStates(cfg)
    print(f"generating task from {start} to {end}")

    dataset = torch.zeros((end - start, cfg['ways'], cfg['shot'] + cfg['queries'], data.shape[2]))
    for iRun in range(end - start):
        dataset[iRun] = GenerateRun(start + iRun, cfg)
    return dataset


if __name__ == "__main__":
    print("Testing Task loader for Few Shot Learning")
    loadDataSet('miniimagenet')

    cfg = {"shot": 1, "ways": 5, "queries": 15}
    setRandomStates(cfg)

    run10 = GenerateRun(10, cfg)
    print("First call:", run10[:2, :2, :2])

    run10 = GenerateRun(10, cfg)
    print("Second call:", run10[:2, :2, :2])

    ds = GenerateRunSet(start=2, end=12, cfg=cfg)
    print("Third call:", ds[8, :2, :2, :2])
    print(ds.size())