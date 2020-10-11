import torch
from itertools import combinations
from torch.distributions import Distribution, Normal, kl_divergence


class GaussianDiagonalMixture(Distribution):
    r"""
    Creates a mixture of diagonal Normal distributions parameterized
    by their means :attr:`means` and scales :attr:`scales`.
    """
    def __init__(self, means, scales):
        assert len(means) == len(scales)
        assert means[0].size(-1) == 1 and scales[0].size(-1) == 1

        self.distributions = []
        for i in range(len(means)):
            self.distributions.append(
                Normal(means[i], scales[i], validate_args=True)
            )

    def expected_mean(self):
        return sum([dist.mean for dist in self.distributions]) / len(self.distributions)

    def expected_entropy(self):
        return sum([dist.entropy().squeeze() for dist in self.distributions]) / len(self.distributions)

    def expected_pairwise_kl(self):
        curr_sum_pairwise_kl = None
        num_pairs = 0

        for dist1, dist2 in combinations(self.distributions, r=2):
            num_pairs += 1
            if curr_sum_pairwise_kl is None:
                curr_sum_pairwise_kl = kl_divergence(dist1, dist2)
            else:
                curr_sum_pairwise_kl += kl_divergence(dist1, dist2)
        return curr_sum_pairwise_kl.squeeze() / num_pairs

    def variance_of_expected(self):
        avg_mean = self.expected_mean()
        return sum([(dist.mean.pow(2) - avg_mean.pow(2)).squeeze() for dist in self.distributions]) / len(self.distributions)

    def log_variance_of_expected(self):
        return self.variance_of_expected().log()

    def expected_variance(self):
        return sum([dist.variance.squeeze() for dist in self.distributions]) / len(self.distributions)

    def log_expected_variance(self):
        return self.expected_variance().log()

    def total_variance(self):
        return self.variance_of_expected() + self.expected_variance()

    def log_total_variance(self):
        return self.total_variance().log()

    def estimated_total_entropy(self):
        return self.expected_entropy() + self.expected_pairwise_kl()

    def log_prob(self, value):
        mean = self.expected_mean()
        var = self.total_variance().unsqueeze(-1)
        return Normal(mean, var.pow(0.5)).log_prob(value)


if __name__ == "__main__":
    ex_means = [torch.ones(32, 1) for _ in range(5)]
    ex_vars = [2 * torch.ones(32, 1) for _ in range(5)]
    mixture_dis = GaussianDiagonalMixture(ex_means, ex_vars)
    print(mixture_dis.expected_mean().shape)
    print(mixture_dis.log_prob(torch.zeros(32, 1)).shape)

    print(mixture_dis.expected_entropy().shape)
    print(mixture_dis.expected_pairwise_kl().shape)
    print(mixture_dis.variance_of_expected().shape)
    print(mixture_dis.expected_variance().shape)
    print(mixture_dis.total_variance().shape)
