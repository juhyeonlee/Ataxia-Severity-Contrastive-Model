import numpy as np
import argparse
import os
import time
import datetime
from contrastive_model import ContrastiveModel
from datautils import load_ftn
from utils import init_dl_program, name_with_datetime
import seaborn as sns
sns.set()


def save_checkpoint_callback(run_dir, save_every=1, unit='epoch'):
    assert unit in ('epoch', 'iter')
    def callback(model, loss):
        n = model.n_epochs if unit == 'epoch' else model.n_iters
        if n % save_every == 0:
            model.save(f'{run_dir}/model_{n}.pkl')
    return callback


def run_train(seg_length_num):
    data_path = 'ftn_no_rotate'
    run_name = 'ftn_no_rotate'
    gpu = 'cpu'
    batch_size = 32
    lr = 0.001
    repr_dims = 320
    iters = None
    epochs = 200
    seed = 42
    max_threads = 8
    freq = 128
    seg_length_sec = 10
    seg_length = freq * 10
    save_every = None

    device = init_dl_program(gpu, seed=seed, max_threads=max_threads, deterministic=False)

    cv_num = 5
    print('Loading data... ', end='')
    train_set = load_ftn(cv_num, seg_length, is_train=True)

    config = dict(batch_size=batch_size,
        lr=lr,
        output_dims=repr_dims,
    )

    run_dir = f"training/{name_with_datetime(run_name)}_{seed}_{seg_length}"
    os.makedirs(run_dir, exist_ok=True)

    if save_every is not None:
        unit = 'epoch' if epochs is not None else 'iter'
        config[f'after_{unit}_callback'] = save_checkpoint_callback(run_dir, save_every, unit)

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

    
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('confignum', metavar='N', type=int, nargs='+',
                    help='an integer for seg_length configuration')

args = parser.parse_args()
run_train(args.confignum[0])