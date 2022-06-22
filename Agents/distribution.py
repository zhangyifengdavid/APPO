from torch.distributions import Categorical
from torch.distributions.kl import kl_divergence

class Distributions():
    def __init__(self, device=None):
        self.device = device if device is not None else 'cpu'

    def sample(self, datas):
        distribution = Categorical(datas)
        return distribution.sample().float().to(self.device)

    def entropy(self, datas):
        distribution = Categorical(datas)
        return distribution.entropy().float().to(self.device)

    def logprob(self, datas, value_data):
        distribution = Categorical(datas)
        return distribution.log_prob(value_data).unsqueeze(1).float().to(self.device)

    def kl_divergence(self, datas1, datas2):
        distribution1 = Categorical(datas1)
        distribution2 = Categorical(datas2)

        return kl_divergence(distribution1, distribution2).unsqueeze(1).float().to(self.device)
