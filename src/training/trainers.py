from typing import Tuple

import torch
from torch.distributions import Distribution
from torch.distributions.kl import kl_divergence
from torch.utils.data import SequentialSampler

from sklearn.metrics import mean_squared_error


def reduce_tensor(vec: torch.Tensor, reduction: str = 'mean'):
    """Global reduction of tensor based on str
    
    Args:
        vec: torch.FloatTensor
        reduction: str, one of ['sum', 'mean', 'none'], default 'mean'
    """
    assert reduction in ['sum', 'mean', 'none']
    if reduction == 'mean':
        return vec.mean()
    elif reduction == 'sum':
        return vec.sum()
    elif reduction == 'none':
        return vec


def params_rmse(predicted_params, targets):
    assert not torch.isnan(predicted_params[0]).any()
    return mean_squared_error(predicted_params[0].cpu(), targets.cpu()) ** 0.5


                

class DistributionMLETrainer:
    """This class implements MLE training for a model that outputs parameters
    of some distribution. Note that both trained model and optimizer instances
    are created inside it.
    """
    def __init__(
        self, model_params: dict, model: torch.nn.Module,
        optim_params: dict, distribution=Distribution,
        optimizer=torch.optim.Adam, scheduler=None, scheduler_params=None,
        test_metrics=[params_rmse], device='cuda:0'
    ):
        self.model = model(**model_params).to(device)
        self.distribution = distribution
        self.optimizer = optimizer(self.model.parameters(), **optim_params)
        self.scheduler = None
        if scheduler is not None:
            self.scheduler = scheduler(self.optimizer, **scheduler_params)
        self.test_metrics = test_metrics
        self.device = device

    @property
    def uncertainty_methods(self):
        return ['entropy']

    def train_step(self, x: torch.FloatTensor, y: torch.FloatTensor) -> float:
        self.optimizer.zero_grad()
        predicted_params = self.model(x)
        loss = self.loss_function(
            predicted_params,
            y
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        return loss.item()

    def eval_step(self, dataloader: torch.utils.data.DataLoader) -> Tuple[float, list]:
        self.model.eval()
        acc_eval_loss = 0.0
        acc_metrics = [0.0 for m in self.test_metrics]
        with torch.no_grad():
            for x, y in dataloader:
                x, y = x.to(self.device), y.to(self.device)
                predicted_params = self.model(x)
                acc_eval_loss += self.loss_function(
                    predicted_params,
                    y
                ).item() / len(dataloader)
                for i, metric in enumerate(self.test_metrics):
                    acc_metrics[i] += metric(
                        predicted_params,
                        y
                    ) / len(dataloader)
        return acc_eval_loss, acc_metrics

    def train(self, dataloader: torch.utils.data.DataLoader,
        num_epochs: int, eval_dataloader: torch.utils.data.DataLoader = None,
        log_per: int = 0, verbose: str =True
    ) -> Tuple[list, list, list]:
        trainloss_hist, valloss_hist, metrics_hist = [], [], []

        for e in range(num_epochs):
            self.model.train()
            acc_train_loss = 0.0
            for x, y in dataloader:
                x, y = x.to(self.device), y.to(self.device)
                acc_train_loss += self.train_step(x, y) / len(dataloader)

            trainloss_hist += [acc_train_loss]

            if eval_dataloader and log_per > 0 and self.test_metrics:
                if e % log_per == 0 or e == (num_epochs - 1):
                    acc_eval_loss, acc_metrics = self.eval_step(eval_dataloader)
                    if verbose:
                        print("Epoch %d train loss %.3f eval loss %.3f" % (
                                e, acc_train_loss, acc_eval_loss
                            ), 'eval ' + ','.join(
                                m.__name__ + " %.3f" % acc_metrics[i]
                                for i, m in enumerate(self.test_metrics)
                            ), flush=True
                        )
                    valloss_hist += [acc_eval_loss]
                    metrics_hist += [acc_metrics]

            if self.scheduler:
                self.scheduler.step()

        return trainloss_hist, valloss_hist, metrics_hist

    def compute_unsupervised_metric_through_data(
        self, dataloader: torch.utils.data.DataLoader, metric
    ) -> torch.FloatTensor:
        metric_scores = []
        self.model.eval()
        with torch.no_grad():
            for items in dataloader:
                if isinstance(items, tuple) or isinstance(items, list):
                    predicted_params = self.model(items[0].to(self.device))
                else:
                    predicted_params = self.model(items.to(self.device))
                metric_scores += metric(
                    predicted_params
                ).cpu().tolist()
        return torch.FloatTensor(metric_scores)

    def loss_function(self, predicted_params, targets):
        return self.nll_loss(predicted_params, targets, reduction='mean')

    def nll_loss(self, predicted_params, targets, reduction='mean'):
        assert reduction in ['mean', 'sum', 'none']
        predicted_dist = self.distribution(*predicted_params)
        batched_loss = -predicted_dist.log_prob(targets)
        assert batched_loss.dim() < 2 or batched_loss.size(-1) == 1
        return reduce_tensor(batched_loss, reduction)

    def get_predicted_params(self, dataloader: torch.utils.data.DataLoader) -> list:
        all_predicted_params = []
        self.model.eval()
        with torch.no_grad():
            for items in dataloader:
                if isinstance(items, tuple) or isinstance(items, list):
                    predicted_params = self.model(items[0].to(self.device))
                else:
                    predicted_params = self.model(items.to(self.device))
                if len(all_predicted_params) == 0:
                    all_predicted_params = [param.cpu().tolist() for param in predicted_params]
                else:
                    for i in range(len(all_predicted_params)):
                        all_predicted_params[i] += predicted_params[i].cpu().tolist()
        return [torch.FloatTensor(param) for param in all_predicted_params]

    def eval_uncertainty(self, dataloader, method: str = 'entropy'):
        some_metric = lambda params: getattr(self.distribution(*params), method)()
        return self.compute_unsupervised_metric_through_data(dataloader, some_metric)

    def save_model(self, dir: str):
        torch.save(self.model.state_dict(), dir + '.ckpt')

    def load_model(self, dir: str):
        self.model.load_state_dict(torch.load(dir + '.ckpt'))


class DistributionRKLTrainer(DistributionMLETrainer):
    """This class replaces standard MLE loss with Reverse-KL.
    """
    def __init__(self, loss_params: dict, *args, **kwargs):
        super(DistributionRKLTrainer, self).__init__(*args, **kwargs)
        self.check_loss_params(loss_params)
        self.loss_params = loss_params

    def loss_function(self, predicted_params, targets):
        target_params = self.converter(targets)
        return self.rkl_loss(predicted_params, target_params, reduction='mean')

    def rkl_loss(self, predicted_params, target_params, reduction='mean'):
        assert reduction in ['mean', 'sum', 'none']
        predicted_dist = self.distribution(*predicted_params)
        true_dist = self.distribution(*target_params)
        batched_loss = kl_divergence(predicted_dist, true_dist)
        assert batched_loss.dim() < 2 or batched_loss.size(-1) == 1
        return reduce_tensor(batched_loss, reduction)

    def converter(self, targets):
        """Extend targets with manually specified params to parametrize target distribution"""
        return [targets] + self.loss_params["real_params"]

    def check_loss_params(self, loss_params):
        for req_key in ['target_lambda', 'target_kappa', 'target_nu']:
            if req_key not in loss_params.keys():
                raise Exception("Rkl loss params dict should contain key", req_key)
        

class DistributionEnsembleMLETrainer:
    """This class sequentially trains multiple models and combines their outputs in an 
    ensemble distribution. Besides more accurate predictions, this allows 
    us to decompose uncertainty measures.
    """
    def __init__(
        self, n_models: int, mixture_distribution=Distribution,
        *args, **kwargs
    ):
        self.trainers = [
            DistributionMLETrainer(*args, **kwargs) for _ in range(n_models)
        ]
        self.mixture_distribution = mixture_distribution

    @property
    def uncertainty_methods(self):
        return [
            'expected_entropy', 'expected_pairwise_kl',
            'variance_of_expected', 'expected_variance', 
            'total_variance'
        ]

    def train(self, dataloader: torch.utils.data.DataLoader,
        num_epochs: int, eval_dataloader: torch.utils.data.DataLoader = None,
        log_per: int = 0, verbose: str =True
    ) -> Tuple[list, list, list]:
        train_hists, val_hists, metrics_hists = [], [], []
        for i, trainer in enumerate(self.trainers):
            if verbose:
                print('-'*20, flush=True)
                print("Model %d" % i, flush=True)
            res = trainer.train(dataloader, num_epochs, eval_dataloader, log_per, verbose)
            train_hists.append(res[0])
            val_hists.append(res[1])
            metrics_hists.append(res[2])
        return train_hists, val_hists, metrics_hists

    def nll_loss(self, predicted_params, targets, reduction='mean'):
        assert reduction in ['mean', 'sum', 'none']
        predicted_dist = self.mixture_distribution(*predicted_params)
        batched_loss = -predicted_dist.log_prob(targets)
        assert batched_loss.dim() < 2 or batched_loss.size(-1) == 1
        return reduce_tensor(batched_loss, reduction)

    def get_predicted_params(self, dataloader: torch.utils.data.DataLoader) -> list:
        if not isinstance(dataloader.sampler, SequentialSampler):
                print(dataloader.batch_sampler)
                raise ValueError("To merge predicted params correctly dataloader shouldn't shuffle")
        all_means, all_stds = [], []
        for trainer in self.trainers:
            cmean, cstd = trainer.get_predicted_params(dataloader)

            all_means.append(cmean)
            all_stds.append(cstd)
        return all_means, all_stds

    def compute_unsupervised_metric_through_data(
        self, dataloader: torch.utils.data.DataLoader, metric
    ) -> torch.FloatTensor:
        metric_scores = []
        for trainer in self.trainers:
            trainer.model.eval()
        with torch.no_grad():
            for items in dataloader:
                all_means, all_stds = [], []
                if isinstance(items, tuple) or isinstance(items, list):
                    for trainer in self.trainers:
                        cmean, cstd = trainer.model(items[0].to(trainer.device))
                        all_means.append(cmean)
                        all_stds.append(cstd)
                else:
                    for trainer in self.trainers:
                        cmean, cstd = trainer.model(items.to(trainer.device))
                        all_means.append(cmean)
                        all_stds.append(cstd)
                if len(all_means[0]) > 1:
                    metric_scores += metric(
                        [all_means, all_stds]
                    ).cpu().tolist()
        return torch.FloatTensor(metric_scores)

    def save_model(self, dir: str):
        for i, trainer in enumerate(self.trainers):
            trainer.save_model(dir + '_' + str(i))

    def load_model(self, dir: str):
        for i, trainer in enumerate(self.trainers):
            trainer.load_model(dir + '_' + str(i))

    def eval_uncertainty(self, dataloader, method: str = 'expected_pairwise_kl'):
        some_metric = lambda params: getattr(self.mixture_distribution(*params), method)()
        return self.compute_unsupervised_metric_through_data(dataloader, some_metric)
