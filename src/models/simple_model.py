import torch
import torch.nn as nn


class GaussianNoise(nn.Module):
    def __init__(self, mean=0.0, sigma=0.05):
        super(GaussianNoise, self).__init__()
        self.mean = mean
        self.sigma = sigma

    def forward(self, input):
        if not self.training:
            return input
        noise = input.clone().normal_(self.mean, self.sigma)
        return input + noise


class SimpleModel(nn.Module):
    def __init__(
        self, input_dim, output_dim, num_units,
        num_hidden=1, activation=nn.ReLU, isPrior=False,
        drop_rate=0.0, use_bn=False, noise_level=0.05
    ):
        super(SimpleModel, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_hidden = num_hidden
        self.isPrior = isPrior
        self.use_bn = use_bn

        # dense network with %num_units hidden layers
        self.features, curr_dim = [], input_dim
        self.features.append(GaussianNoise(sigma=noise_level))
        for _ in range(num_hidden):
            self.features.append(nn.Linear(curr_dim, num_units))
            if self.use_bn:
                self.features.append(nn.BatchNorm1d(num_units))
            self.features.append(activation())
            if drop_rate > 0.0:
                self.features.append(nn.Dropout(drop_rate))
            curr_dim = num_units
        self.features = nn.Sequential(*self.features)

        # generate stats of output distribution
        self.layer_mean = nn.Linear(num_units, output_dim)
        self.layer_std = nn.Sequential(
            nn.Linear(num_units, output_dim),
            nn.Softplus()
        )
        if isPrior:
            self.layer_beta = nn.Sequential(
                nn.Linear(num_units, 1),
                nn.Softplus()
            )

        self._initialize_weights()

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        x = self.features(x)

        mean = self.layer_mean(x)
        std = self.layer_std(x) + 1e-6
        if self.isPrior:
            beta = self.layer_beta(x) + 1e-6
            kappa = beta
            nu = beta + self.output_dim + 1
            return mean, std, kappa[..., 0], nu[..., 0]
        else:
            return mean, std

    def _initialize_weights(self):
        """Initialize weights as in 
        `Probabilistic Backpropagation for Scalable Learning of Bayesian Neural Networks`
        (https://arxiv.org/pdf/1502.05336.pdf), section 3.5
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 1.0 / (m.weight.size(1) + 1))
                nn.init.constant_(m.bias, 0)
