from contextlib import ExitStack

import torch


class Adam(object):

    def __init__(self, lr, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

    def __call__(self, step, params, grads, means, variances, bp_through_optimizer=False):
        with torch.no_grad() if not bp_through_optimizer else ExitStack():
            new_params = []
            new_means = []
            new_variances = []
            for param, grad, mean, variance in zip(params, grads, means, variances):
                # if not bp_through_optimizer:
                #     mean = mean.detach()
                #     variance = variance.detach()
                #     grad = grad.detach()
                new_means.append(self.beta1 * mean + (1 - self.beta1) * grad)
                new_variances.append(self.beta2 * variance + (1 - self.beta2) * grad * grad)

                c_m = new_means[-1] / (1 - self.beta1 ** step)
                c_v = new_variances[-1] / (1 - self.beta2 ** step)
                new_params.append(param - self.lr / (torch.sqrt(c_v) + self.eps) * c_m)

        return new_params, new_means, new_variances
