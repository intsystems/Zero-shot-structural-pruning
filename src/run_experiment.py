from __future__ import annotations

import json
import os

import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
from collections import defaultdict
from sklearn.linear_model import LinearRegression

import hydra
from omegaconf import DictConfig, OmegaConf

from tqdm import trange

from utils import update_log, get_data, train
from model.random_graph import RandomGraphModel, generate_graph
from surrogate.dataset import create_surrogate_dataset
from surrogate.graph_model import SurrogateModel
from surrogate.fcn_model import create_surrogate_fcn_model
from surrogate.test_models import test_models, collect_model_stats, get_expected_intersection


def resolve_device(device_cfg: str) -> str:
    if device_cfg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_cfg


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    # Resolve device
    device = resolve_device(cfg.device)

    # Optional seed
    if cfg.seed is not None:
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

    # Data loaders
    train_loader, test_loader = get_data(
        batch_size=cfg.model.batch_size_mnist,
        data_dir=cfg.data.data_dir,
        num_workers=cfg.data.num_workers,
    )

    mnist_losses_history = []
    surrogate_losses_history = []
    fcn_losses_history = []
    metric_results = defaultdict(list)
    preds_results = defaultdict(list)
    y_true_all = []

    # top-K intersection metrics (top-5% and top-10% of n_iterations_masks)
    n_models = cfg.experiment.n_iterations_masks
    n_models_5 = int(n_models * 0.05)
    n_models_10 = int(n_models * 0.1)
    topk_results: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])

    tbar = trange(cfg.experiment.n_iterations_global, desc='Global iteration')
    for i in tbar:
        update_log(i, 'global')

        # Sampling and creating graph model
        modules, edges = generate_graph(
            cfg.model.input_dim,
            cfg.experiment.n_iterations_graph,
        )
        model = RandomGraphModel(modules, edges)

        # Training model on MNIST
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.model.lr_mnist)
        mnist_losses_history.append(train(
            model, train_loader, criterion, optimizer,
            device, cfg.model.n_epochs_mnist,
            'MNIST training', is_surrogate=False,
        ))

        # Creating a dataset for surrogate model
        surrogate_dataset_train, surrogate_dataset_test = create_surrogate_dataset(
            model, test_loader, nn.CrossEntropyLoss(),
            cfg.experiment.n_iterations_masks,
            cfg.experiment.mask_prob,
            device,
        )
        surrogate_train_loader = DataLoader(
            surrogate_dataset_train,
            batch_size=cfg.surrogate.batch_size_surrogate,
            shuffle=True,
        )

        # Calculating L1 and Molchanov importances
        importance_stats = collect_model_stats(model, test_loader, criterion, optimizer, device)

        # Creating and training surrogate graph model
        surrogate_graph_model = SurrogateModel(edges)
        criterion_surrogate = nn.MSELoss()
        optimizer_surrogate = torch.optim.Adam(
            surrogate_graph_model.parameters(), lr=cfg.surrogate.lr_surrogate
        )
        surrogate_losses_history.append(train(
            surrogate_graph_model, surrogate_train_loader,
            criterion_surrogate, optimizer_surrogate, device,
            cfg.surrogate.n_epochs_surrogate,
            'Surrogate graph training', is_surrogate=True,
        ))

        # Creating and training surrogate FCN model
        surrogate_fcn_model = create_surrogate_fcn_model(
            len(edges), list(cfg.surrogate.fcn_dims), add_sigmoid=False
        )
        criterion_fcn = nn.MSELoss()
        optimizer_fcn = torch.optim.Adam(
            surrogate_fcn_model.parameters(), lr=cfg.surrogate.lr_fcn
        )
        fcn_losses_history.append(train(
            surrogate_fcn_model, surrogate_train_loader,
            criterion_fcn, optimizer_fcn, device,
            cfg.surrogate.n_epochs_fcn,
            'Surrogate FCN training', is_surrogate=True,
        ))

        # Linear model
        surrogate_linear_model = LinearRegression()
        surrogate_linear_model.fit(
            surrogate_dataset_train.tensors[0].numpy(),
            surrogate_dataset_train.tensors[1].numpy(),
        )

        # Testing models
        name2model = {
            'Graph': surrogate_graph_model,
            'FCN': surrogate_fcn_model,
            'Linear': surrogate_linear_model,
            'L1': importance_stats['l1'],
            'Molchanov': importance_stats['molchanov'],
        }

        name2metric, name2preds, y_true = test_models(name2model, surrogate_dataset_test, device)
        y_true_all.append(y_true.tolist())

        y_true_np = np.array(y_true.tolist())
        true_top5 = np.argsort(y_true_np, stable=True)[-n_models_5:]
        true_top10 = np.argsort(y_true_np, stable=True)[-n_models_10:]

        for name_model in name2model:
            metric_results[name_model].append(name2metric[name_model])
            preds_list = (
                name2preds[name_model] if isinstance(name2preds[name_model], list)
                else name2preds[name_model].tolist()
            )
            preds_results[name_model].append(preds_list)

            preds_np = np.array(preds_list)
            exp_5 = get_expected_intersection(true_top5, preds_np, n_models_5)
            exp_10 = get_expected_intersection(true_top10, preds_np, n_models_10)
            topk_results[name_model][0] += exp_5 / cfg.experiment.n_iterations_global
            topk_results[name_model][1] += exp_10 / cfg.experiment.n_iterations_global

    # Save results
    n_iter = cfg.experiment.n_iterations_graph
    os.makedirs(cfg.output.output_dir, exist_ok=True)
    history_to_save = {
        "config": OmegaConf.to_container(cfg, resolve=True),
        "mnist_losses_history": mnist_losses_history,
        "surrogate_losses_history": surrogate_losses_history,
        "fcn_losses_history": fcn_losses_history,
        "metric_results": dict(metric_results),
        "preds_results": dict(preds_results),
        "y_true_all": y_true_all,
        "topk_results": dict(topk_results),
    }
    filename = os.path.join(cfg.output.output_dir, f"history_{n_iter}_iterations.json")
    with open(filename, 'w') as f:
        json.dump(history_to_save, f)

    print(f"\nResults saved to {filename}")
    print("\nMean RMSE per model:")
    for model_name, metrics in metric_results.items():
        if model_name not in ('L1', 'Molchanov'):
            print(f"  {model_name}: {sum(metrics) / cfg.experiment.n_iterations_global:.2f}%")

    print(f"\nMean top-K intersection (top-5% = {n_models_5}, top-10% = {n_models_10}):")
    print(f"  {'Model':<12} {'top-5%':>8} {'top-10%':>8}")
    for model_name, (t5, t10) in topk_results.items():
        print(f"  {model_name:<12} {t5:>8.2f} {t10:>8.2f}")


if __name__ == '__main__':
    main()
