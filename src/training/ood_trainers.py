from typing import Tuple
from itertools import cycle

import torch
from torch.distributions import Distribution
import torch.nn as nn
from torch.optim import Adam

from src.training.trainers import DistributionRKLTrainer
from src.utils.func_utils import reduce_tensor, params_rmse


class DistributionRKLTrainerWithOOD(DistributionRKLTrainer):
    @property
    def uncertainty_methods(self):
        return [
            'predictive_posterior_entropy', 'expected_entropy',
            'mutual_information', 'expected_pairwise_kl',
            'variance_of_expected', 'expected_variance',
            'total_variance'
        ]

    def train_step(self, x, y, x_ood):
        self.optimizer.zero_grad()

        predicted_params = self.model(x)
        prior_params = self.prior_converter(x)

        ordinary_loss = self.loss_function(
            predicted_params,
            prior_params,
            y
        )
        
        self.switch_bn_updates("eval")
        predicted_ood_params = self.model(x_ood)
        prior_params = self.prior_converter(x_ood)

        ood_loss = self.loss_function(
            predicted_ood_params,
            prior_params
        )
        self.switch_bn_updates("train")
        loss = ordinary_loss + self.loss_params["ood_coeff"] * ood_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        return ordinary_loss.item(), self.loss_params["ood_coeff"] * ood_loss.item()

    def loss_function(self, predicted_params, prior_params, targets=None):
        if targets is None:
            return self.loss_params["inv_real_beta"] * self.rkl_loss(
                predicted_params, prior_params, reduction='mean'
            )
        else:
            predicted_dist = self.distribution(*predicted_params)
            inv_beta = self.loss_params["inv_real_beta"]
            return -predicted_dist.expected_log_prob(targets).mean() + inv_beta * self.rkl_loss(
                predicted_params, prior_params, reduction='mean'
            )

    def eval_step(self, dataloader: torch.utils.data.DataLoader) -> Tuple[float, list]:
        self.model.eval()
        acc_eval_loss = 0.0
        acc_metrics = [0.0 for m in self.test_metrics]
        with torch.no_grad():
            for i, (x, y) in enumerate(dataloader):
                x, y = x.to(self.device), y.to(self.device)
                predicted_params = self.model(x)
                prior_params = self.prior_converter(x)
                acc_eval_loss += self.loss_function(
                    predicted_params,
                    prior_params,
                    y
                ).item() / len(dataloader)
                for i, metric in enumerate(self.test_metrics):
                    acc_metrics[i] += metric(
                        predicted_params,
                        y
                    ) / len(dataloader)
        return acc_eval_loss, acc_metrics

    def train(self, dataloader, oodloader, num_epochs, eval_dataloader=None, log_per=0, verbose=True):
        with torch.no_grad():
            self.estimate_avg_mean_var(dataloader)

        trainloss_hist, oodloss_hist, valloss_hist, metrics_hist = [], [], [], []

        for e in range(num_epochs):
            self.model.train()
            acc_train_loss = 0.0
            acc_ood_loss = 0.0
            """With only zip() the iterator will be exhausted when the length
            is equal to that of the smallest dataset.
            But with the use of cycle(), we will repeat the smallest dataset again unless
            our iterator looks at all the samples from the largest dataset."""
            for (x, y), (x_ood,) in zip(dataloader, cycle(oodloader)):
                x, y, x_ood = x.to(self.device), y.to(self.device), x_ood.to(self.device)
                c_losses = self.train_step(x, y, x_ood)
                acc_train_loss += c_losses[0] / len(dataloader)
                acc_ood_loss += c_losses[1] / len(dataloader)

            trainloss_hist += [acc_train_loss]
            oodloss_hist += [acc_ood_loss]

            if eval_dataloader and log_per > 0 and self.test_metrics:
                if e % log_per == 0 or e == (num_epochs - 1):
                    acc_eval_loss, acc_metrics = self.eval_step(eval_dataloader)

                    if verbose:
                        print("Epoch %d train loss %.3f ood loss %.3f eval loss %.3f" % (
                                e, acc_train_loss, acc_ood_loss, acc_eval_loss
                            ), 'eval ' + ','.join(m.__name__ + " %.3f" % acc_metrics[i] for i, m in enumerate(self.test_metrics)),
                            flush=True
                        )
                    valloss_hist += [acc_eval_loss]
                    metrics_hist += [acc_metrics]

            if self.scheduler:
                self.scheduler.step()

        return trainloss_hist, oodloss_hist, valloss_hist, metrics_hist

    def estimate_avg_mean_var(self, dataloader):
        self.avg_mean = None
        for _, y in dataloader:
            if self.avg_mean is None:
                self.avg_mean = y.mean(dim=0) / len(dataloader)
            else:
                self.avg_mean += y.mean(dim=0) / len(dataloader)
        sum_var = torch.zeros_like(self.avg_mean)
        num_samples = 0
        for _, y in dataloader:
            avg_mean = torch.repeat_interleave(self.avg_mean.unsqueeze(0), repeats=y.size(0), dim=0)
            sum_var += (y - avg_mean).pow(2).sum(dim=0)
            num_samples += y.size(0)
        self.avg_scatter = sum_var / num_samples

    def prior_converter(self, inputs):
        avg_mean_r = torch.repeat_interleave(
            self.avg_mean.unsqueeze(0), repeats=inputs.size(0), dim=0
        ).to(inputs.device)
        prior_kappa, prior_nu = self.loss_params['prior_beta'],\
            self.loss_params['prior_beta'] + self.model.output_dim + 1
        avg_precision_r = torch.repeat_interleave(
            (1 / (prior_nu * self.avg_scatter.unsqueeze(0))), repeats=inputs.size(0), dim=0
        ).to(inputs.device)

        all_params = [avg_mean_r, avg_precision_r]
        return all_params + [prior_kappa, prior_nu]

    def nll_loss(self, predicted_params, targets, reduction='mean'):
        assert reduction in ['mean', 'sum', 'none']
        predicted_dist = self.distribution(*predicted_params)
        batched_loss = -predicted_dist.predictive_posterior_log_prob(targets)
        assert batched_loss.dim() < 2 or batched_loss.size(-1) == 1
        return reduce_tensor(batched_loss, reduction)

    def switch_bn_updates(self, mode):
        if self.model.use_bn:
            for m in self.model.modules():
                if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
                    if mode == 'train':
                        m.train()
                    elif mode == 'eval':
                        m.eval()

    def check_loss_params(self, loss_params):
        for req_key in ['inv_real_beta', 'ood_coeff', 'prior_beta']:
            if req_key not in loss_params.keys():
                raise Exception("Rkl loss params dict should contain key", req_key)


class DistributionEnsembleToPriorDistiller(DistributionRKLTrainer):
    def __init__(
        self, teacher_models: list, *args, **kwargs
    ):
        super(DistributionEnsembleToPriorDistiller, self).__init__(*args, **kwargs)
        self.teacher_models = teacher_models
        self.num_steps = 1
        for model in self.teacher_models:
            model.eval()
        self.loss_params['temperature'] = self.loss_params['max_temperature']

    @property
    def uncertainty_methods(self):
        return [
            'predictive_posterior_entropy', 'expected_entropy',
            'mutual_information', 'expected_pairwise_kl',
            'variance_of_expected', 'expected_variance',
            'total_variance'
        ]

    def train_step(self, x, y):
        x += torch.empty(x.shape).normal_(
            mean=0, std=self.loss_params["noise_level"]
        ).to(x.device)

        if "max_steps" in self.loss_params.keys():
            T_0 = self.loss_params["max_temperature"]
            first_part = float(0.2 * self.loss_params["max_steps"])
            third_part = float(0.6 * self.loss_params["max_steps"])
            if self.num_steps < first_part:
                self.loss_params["temperature"] = T_0
            elif self.num_steps < third_part:
                self.loss_params["temperature"] = T_0 - (T_0 - 1) * min(
                    float(self.num_steps - first_part) / float(0.4 * self.loss_params["max_steps"]),
                    1.0
                )
            else:
                self.loss_params["temperature"] = 1.0

        self.optimizer.zero_grad()

        predicted_params = self.model(x)
        loss = self.loss_function(
            predicted_params,
            x
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

        self.optimizer.step()
        self.num_steps += 1
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
                    x
                ).item() / len(dataloader)
                for i, metric in enumerate(self.test_metrics):
                    acc_metrics[i] += metric(
                        predicted_params,
                        y
                    ) / len(dataloader)
        return acc_eval_loss, acc_metrics

    def loss_function(self, predicted_params, x):
        T = self.loss_params["temperature"]
        with torch.no_grad():
            all_teachers_means, all_teachers_vars = [], []
            aggr_teachers_mean = torch.zeros_like(predicted_params[0]).to(self.device)
            aggr_teachers_var = torch.zeros_like(predicted_params[1]).to(self.device)
            for i, teacher in enumerate(self.teacher_models):
                teacher_params = teacher(x)
                aggr_teachers_mean += teacher_params[0] / len(self.teacher_models)
                aggr_teachers_var += teacher_params[1].pow(2) / len(self.teacher_models)
                all_teachers_means.append(teacher_params[0])
                all_teachers_vars.append(teacher_params[1].pow(2))
            for i, _ in enumerate(self.teacher_models):
                all_teachers_means[i] = (T - 1) * aggr_teachers_mean / (T + 1) +\
                    2 * all_teachers_means[i] / (T + 1)
                all_teachers_vars[i] = (T - 1) * aggr_teachers_var / (T + 1) +\
                    2 * all_teachers_vars[i] / (T + 1)

        new_nu = (predicted_params[-1] - self.model.output_dim - 1) * T +\
            self.model.output_dim + 1
        new_kappa = predicted_params[-2] * T
        #new_nu = (predicted_params[-1] - self.model.output_dim - 2) * (1 / T) +\
        #    self.model.output_dim + 2
        #new_kappa = predicted_params[-2] * (1 / T)

        predicted_dist = self.distribution(*[
            predicted_params[0], predicted_params[1], new_kappa, new_nu
        ])
        all_losses = []
        for i in range(len(self.teacher_models)):
            all_losses.append(-predicted_dist.log_prob(
                all_teachers_means[i], 1 / all_teachers_vars[i]
            ).sum())

        return sum(all_losses) / len(all_losses) / T

    def nll_loss(self, predicted_params, targets, reduction='mean'):
        assert reduction in ['mean', 'sum', 'none']
        predicted_dist = self.distribution(*predicted_params)
        batched_loss = -predicted_dist.predictive_posterior_log_prob(targets)
        assert batched_loss.dim() < 2 or batched_loss.size(-1) == 1
        return reduce_tensor(batched_loss, reduction)

    def check_loss_params(self, loss_params):
        for req_key in ['max_temperature', 'noise_level']:
            if req_key not in loss_params.keys():
                raise Exception("NLL loss params dict should contain key", req_key)
