import math, torch

from torch.distributions import StudentT
from .distributions import NormalDiagonalWishart
from src.utils.func_utils import mvdigamma


class NormalWishartPrior(NormalDiagonalWishart):

    def forward(self):
        self.precision_coeff = (self.belief + 1) / (
            self.belief * (self.df - self.dimensionality + 1)
        )
        return StudentT(
            (self.df - self.dimensionality + 1).unsqueeze(-1),
            loc=self.loc,
            scale=(self.precision_coeff.unsqueeze(-1) / self.precision_diag).pow(0.5),
        )

    def predictive_posterior_log_prob(self, value):
        return self.forward().log_prob(value)

    def predictive_posterior_variance(self):
        variance_res = self.forward().variance
        if variance_res.size(-1) != 1:
            raise ValueError("Predictive posterior returned entropy with incorrect shapes")
        return variance_res[..., 0]

    def log_predictive_posterior_variance(self):
        return self.predictive_posterior_variance().log()

    def predictive_posterior_entropy(self):
        entropy_res = self.forward().entropy()
        if entropy_res.size(-1) != 1:
            raise ValueError("Predictive posterior returned entropy with incorrect shapes")
        return entropy_res[..., 0]

    def entropy_ub(self):
        return self.expected_pairwise_kl() + self.expected_entropy()

    def expected_entropy(self):
        mvdigamma_term = mvdigamma(0.5 * self.df, self.dimensionality)
        return 0.5 * (
            self.dimensionality * (1 + math.log(2 * math.pi))
            - (2 * self.precision_diag).log().sum(dim=-1)
            - mvdigamma_term.squeeze()
        )

    def expected_log_prob(self, value):
        neg_mse_term = -torch.sum(
            (self.loc - value).pow(2) * self.precision_diag * self.df.unsqueeze(-1),
            dim = -1
        )
        mvdigamma_term = mvdigamma(0.5 * self.df, self.dimensionality)

        reg_terms = (2 * self.precision_diag).log().sum(dim=-1) + mvdigamma_term
        conf_term = -self.dimensionality * self.belief.pow(-1)
        return 0.5 * (neg_mse_term + reg_terms + conf_term)

    def mutual_information(self):
        predictive_posterior_entropy = self.predictive_posterior_entropy()
        expected_entropy = self.expected_entropy()
        return predictive_posterior_entropy - expected_entropy

    def expected_pairwise_kl(self):
        term1 = 0.5 * (
            self.df * self.dimensionality / (self.df - self.dimensionality - 1) -\
                self.dimensionality
        )
        term2 = 0.5 * (
            self.df * self.dimensionality / (self.df - self.dimensionality - 1) +\
                self.dimensionality
        ) / self.belief
        return term1 + term2

    def variance_of_expected(self):
        return self.expected_variance() / self.belief

    def log_variance_of_expected(self):
        return self.variance_of_expected().log()

    def expected_variance(self):
        result = 1 / (self.precision_diag * (self.df.unsqueeze(-1) - self.dimensionality - 1))
        if result.size(-1) != 1:
            raise ValueError("Expected variance currently supports only one-dimensional targets")

        return result[..., 0]

    def log_expected_variance(self):
        return self.expected_variance().log()

    def total_variance(self):
        tv = self.variance_of_expected() + self.expected_variance()
        ppv = self.predictive_posterior_variance()

        rel_diff = (tv - ppv).abs() / tv.abs().pow(0.5) / ppv.abs().pow(0.5)
        assert (rel_diff < 1e-6).all()
        return tv

    def log_total_variance(self):
        return self.total_variance().log()


if __name__ == '__main__':
    ex_mean = torch.zeros(32, 200, 400, 1)
    ex_var = torch.ones(32, 200, 400, 1)
    ex_belief = torch.ones(32, 200, 400)
    ex_df = 10 * torch.ones(32, 200, 400)

    ex_dist = NormalWishartPrior(ex_mean, ex_var, ex_belief, ex_df)
    print(ex_dist.predictive_posterior_log_prob(2 * torch.ones(32, 200, 400, 1)).shape)
    print(ex_dist.log_prob(2 * torch.ones(32, 200, 400, 1), 2 * torch.ones(32, 200, 400, 1)).shape)

    print(ex_dist.predictive_posterior_entropy().shape) #Total
    print(ex_dist.expected_entropy().shape) #Data
    print(ex_dist.mutual_information().shape) #Knowledge
    print(ex_dist.expected_pairwise_kl().shape) #Knowledge
    print(ex_dist.variance_of_expected().shape) #Knowledge
    print(ex_dist.expected_variance().shape) #Data
    print(ex_dist.total_variance().shape) #Total
    print(ex_dist.predictive_posterior_variance().shape) #Total

