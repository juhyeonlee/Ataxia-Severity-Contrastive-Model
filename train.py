import numpy as np
import argparse
import os
import time
import datetime
from contrastive_model import ContrastiveModel
from datautils import load_ftn
from utils import init_dl_program, name_with_datetime
import seaborn as sns
import yaml
sns.set()


def run_train(config_path):
    with open(config_path, 'r') as file:
        cfg = yaml.safe_load(file)

    run_name = cfg.get('run_name', 'ftn_no_rotate')
    gpu = cfg.get('gpu', '0')
    batch_size = cfg.get('batch_size', 32)
    lr = cfg.get('lr', 0.001)
    repr_dims = cfg.get('repr_dims', 320)
    epochs = cfg.get('epochs', 200)
    seed = cfg.get('seed', 42)
    max_threads = cfg.get('max_threads', 8)
    freq = cfg.get('freq', 128)
    seg_length_sec = cfg.get('seg_length_sec', 10)
    cv_num = cfg.get('cv_num', 5)
    seg_length = freq * seg_length_sec

    device = init_dl_program(gpu, seed=seed, max_threads=max_threads, deterministic=False)

    print('Loading data... ', end='')
    train_set = load_ftn(cv_num, seg_length, is_train=True)

    config = dict(batch_size=batch_size, lr=lr, output_dims=repr_dims)

    run_dir = f"training/{name_with_datetime(run_name)}_{seed}_{seg_length}"
    os.makedirs(run_dir, exist_ok=True)

    for cv in range(cv_num):
        t = time.time()
        contrast_weight = train_set[cv]['total_bars'].copy()

        model = ContrastiveModel(
            input_dims=train_set[cv]['data'].shape[-1],
            device=device,
            **config
        )
        loss_train_log = model.fit(
            train_set[cv]['data'],
            contrast_weight,
            n_epochs=epochs,
            verbose=True,
        )
        model.save(f"{run_dir}/model_{cv}.pkl")

        t = time.time() - t
        print(f"\nTraining time: {datetime.timedelta(seconds=t)}\n")
    print("Finished.")

# Example usage
# run_train(config_path='config.yaml')