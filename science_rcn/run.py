"""
Reference implementation of a two-level RCN model for MNIST classification experiments.

Examples:
- To run a small unit test that trains and tests on 20 images using one CPU 
  (takes ~2 minutes, accuracy is ~60%):
python science_rcn/run.py

- To run a slightly more interesting experiment that trains on 100 images and tests on 20 
  images using multiple CPUs (takes <1 min using 7 CPUs, accuracy is ~90%):
python science_rcn/run.py --train_size 100 --test_size 20 --parallel

- To test on the full 10k MNIST test set, training on 1000 examples 
(could take hours depending on the number of available CPUs, average accuracy is ~97.7+%):
python science_rcn/run.py --full_test_set --train_size 1000 --parallel --pool_shape 25 --perturb_factor 2.0
"""

import os
import argparse
import numpy as np
from PIL import Image
from functools import partial
from multiprocessing import Pool
from alisuretool.Tools import Tools
from science_rcn.inference import test_image
from science_rcn.learning import train_image


def get_mnist_data(data_dir, train_size, test_size, full_test_set=False, seed=5):

    def _load_data(image_dir, num_per_class):
        loaded_data = []
        for category in sorted(os.listdir(image_dir)):
            cat_path = os.path.join(image_dir, category)
            if not os.path.isdir(cat_path) or category.startswith('.'):
                continue
            samples = sorted(os.listdir(cat_path))
            if num_per_class is not None:
                samples = np.random.choice(samples, num_per_class)

            for sample in samples:
                image_arr = np.array(Image.open(os.path.join(cat_path, sample)).resize((112, 112)))
                image_arr = np.pad(image_arr, tuple([(p, p) for p in (44, 44)]), mode='constant', constant_values=0)
                loaded_data.append((image_arr, category))
                pass
            pass
        return loaded_data

    np.random.seed(seed)
    train_set = _load_data(os.path.join(data_dir, 'training'), num_per_class=train_size//10)
    test_set = _load_data(os.path.join(data_dir, 'testing'), num_per_class=None if full_test_set else test_size//10)
    return train_set, test_set


def run_experiment(data_dir='..\\data\\MNIST', train_size=20, test_size=20,
                   full_test_set=False, pool_shape=(25, 25), perturb_factor=2., parallel=True, seed=5):
    # Data
    train_data, test_data = get_mnist_data(data_dir, train_size, test_size, full_test_set, seed=seed)
    pool = Pool(None if parallel else 1)

    # Training
    Tools.print("Training on {} images...".format(len(train_data)))
    train_partial = partial(train_image, perturb_factor=perturb_factor)
    train_results = pool.map_async(train_partial, [d[0] for d in train_data]).get(99999)
    all_model_factors = zip(*train_results)

    # Testing
    Tools.print("Testing on {} images...".format(len(test_data)))
    test_partial = partial(test_image, model_factors=list(all_model_factors), pool_shape=pool_shape)
    test_results = pool.map_async(test_partial, [d[0] for d in test_data]).get(99999)

    # Evaluate result
    correct = [int(test_data[idx][1])==winner//(train_size//10) for idx, (winner, _) in enumerate(test_results)]
    Tools.print("Total test accuracy = {}".format(float(sum(correct)) / len(test_results)))
    pass


if __name__ == '__main__':
    _train_size, _test_size = 20, 20
    _full_test_set, _parallel = False, True
    _pool_shape, _perturb_factor = 25, 2.
    run_experiment(train_size=_train_size, test_size=_test_size, full_test_set=_full_test_set,
                   parallel=_parallel, pool_shape=(_pool_shape, _pool_shape), perturb_factor=_perturb_factor)
