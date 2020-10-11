import pandas as pd
import numpy as np
import torch
from torch.utils import data as tdata


def get_toy_dataset(
    target_generator_fn,
    noise_generator_fn,
    train_limits=(-1.0, 1.0),
    test_limits=(-1.5, 1.5), ood_abs_limits=(1.1, 1.3),
    train_samples=100, test_samples=200,
    ood_samples=40, random_state=0
):
    """Generates one-dimensional regression dataset"""
    np.random.seed(random_state)
    x_train = np.random.uniform(train_limits[0], train_limits[1], (train_samples,))

    y_train = target_generator_fn(x_train)
    np.random.seed(random_state)
    y_noise = noise_generator_fn(x_train) * np.random.randn(y_train.shape[0])
    y_train += y_noise

    np.random.seed(random_state)
    x_ood_1 = np.random.uniform(ood_abs_limits[0], ood_abs_limits[1], (ood_samples // 2,))
    np.random.seed(random_state)
    x_ood_2 = np.random.uniform(-ood_abs_limits[1], -ood_abs_limits[0], (ood_samples // 2,))
    x_ood = np.concatenate([x_ood_1, x_ood_2], axis=0)

    y_ood = target_generator_fn(x_ood)
    np.random.seed(random_state)
    y_ood += noise_generator_fn(x_ood) * np.random.randn(y_ood.shape[0])

    x_test = np.linspace(test_limits[0], test_limits[1], test_samples)
    y_test = target_generator_fn(x_test)

    train_data, test_data, ood_data = [
        tdata.TensorDataset(
            torch.Tensor(x_c).unsqueeze(1),
            torch.Tensor(y_c).unsqueeze(1)
        ) for (x_c, y_c) in zip(
            [x_train, x_test, x_ood], [y_train, y_test, y_ood]
        )
    ]
    return train_data, test_data, ood_data, y_noise

def get_arrays_from_loader(loader):
    first_elems = []
    second_elems = []
    for item in loader:
        first_elems += [item[0]]
        second_elems += [item[1]]
    return torch.cat(first_elems, dim=0), torch.cat(second_elems, dim=0)

def get_table_loaders(
    train_data, test_data, batch_size, ood_data=None, ood_test_data=None,
    ood_batch_size=None, shuffle=True, normalize_targets=False, target_id=-1,
):
    feature_len = train_data.shape[1] - 1
    if target_id == -1:
        x_train, y_train = train_data[:, :-1], train_data[:, -1:]
        x_test, y_test = test_data[:, :-1], test_data[:, -1:]
    elif target_id > -1:
        idxs = list(range(train_data.shape[1]))
        idxs.pop(target_id)
        x_train, y_train = train_data[:, idxs],\
            train_data[:, target_id].reshape(-1,1)
        x_test, y_test = test_data[:, idxs],\
            test_data[:, target_id].reshape(-1,1)
    else:
        raise ValueError("Provide target_id >= -1")

    # Normalize train/test features & targets (if necessary)
    x_means, x_stds = x_train.mean(axis=0), x_train.std(axis=0)
    if normalize_targets:
        y_means, y_stds = y_train.mean(axis=0), y_train.std(axis=0)
    else:
        y_means, y_stds = 0., 1.
    x_train = (x_train - x_means) / x_stds
    y_train = (y_train - y_means) / y_stds
    x_test = (x_test - x_means) / x_stds
    y_test = (y_test - y_means) / y_stds
    # Normalize ood features
    if ood_data is not None:
        if target_id > -1:
            idxs = list(range(train_data.shape[1]))
            idxs.pop(target_id)
            x_ood = ood_data[:, idxs]
            x_ood = (x_ood - x_means) / x_stds
        else: 
            x_ood = ood_data[:, :feature_len]
            x_ood = (x_ood - x_means) / x_stds
        assert not np.isnan(x_ood).any()
    if ood_test_data is not None:
        if target_id > -1:
            idxs = list(range(train_data.shape[1]))
            idxs.pop(target_id)
            x_ood_test = ood_test_data[:, idxs]
            x_ood_test = (x_ood_test - x_means) / x_stds
        else: 
            x_ood_test = ood_test_data[:, :feature_len]
            x_ood_test = (x_ood_test - x_means) / x_stds
        assert not np.isnan(x_ood_test).any()

    assert not np.isnan(y_test).any()
    assert not np.isnan(y_train).any()
    assert not np.isnan(x_test).any()
    assert not np.isnan(x_train).any()
    ood_loader = None
    ood_test_loader = None
    # Initialize loaders
    train_loader = tdata.DataLoader(
        tdata.TensorDataset(
            torch.Tensor(x_train), torch.Tensor(y_train)
        ),
        batch_size=batch_size,
        shuffle=shuffle
    )
    test_loader = tdata.DataLoader(
        tdata.TensorDataset(
            torch.Tensor(x_test), torch.Tensor(y_test)
        ),
        batch_size=batch_size, shuffle=False
    )
    if ood_data is not None:
        ood_loader = tdata.DataLoader(
            tdata.TensorDataset(torch.Tensor(x_ood)),
            batch_size=ood_batch_size, shuffle=shuffle
        )
    if ood_test_data is not None:
        ood_test_loader = tdata.DataLoader(
            tdata.TensorDataset(torch.Tensor(x_ood_test)),
            batch_size=ood_batch_size, shuffle=False
        )
    return train_loader, test_loader, ood_loader, ood_test_loader,\
        [torch.FloatTensor([y_means]), torch.FloatTensor([y_stds])]
