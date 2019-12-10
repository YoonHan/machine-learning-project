import torch
import math
from torch.optim.optimizer import Optimizer


class ADAMOptimizer(Optimizer):
    """
        ADAM = Adagrad + RMSprop
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(ADAMOptimizer, self).__init__(params, defaults)

    def step(self):
        """
            optimisation 한 스텝을 수행한다.
        """
        loss = None
        for group in self.param_groups:
            for p in group['params']:
                grad = p.grad.data
                state = self.state[p]

                # initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Momentum (gradient value 들의 지수가중평균)
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # RMS Prop (squared gradient value 들의 지수가중평균). 분모에 들어간다.
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                b1, b2 = group['betas']
                state['step'] += 1

                # L2 penalty. Gotta add to Gradient as well.
                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                # Momentum
                exp_avg = torch.mul(exp_avg, b1) + (1 - b1)*grad
                # RMS Prop
                exp_avg_sq = torch.mul(exp_avg_sq, b2) + (1-b2)*(grad*grad)

                denominator = exp_avg_sq.sqrt() + group['eps']

                bias_correction1 = 1 / (1 - b1 ** state['step'])
                bias_correction2 = 1 / (1 - b2 ** state['step'])

                adaptive_learning_rate = group['lr'] * bias_correction1 / math.sqrt(bias_correction2)

                p.data = p.data - adaptive_learning_rate * exp_avg / denominator

                if state['step']  % 10000 ==0:
                    print ("group:", group)
                    print("p: ",p)
                    print("p.data: ", p.data) # W = p.data

        return loss