from torch.optim import lr_scheduler
from typing import List
from bisect import bisect_right


class WarmupStepLR(lr_scheduler._LRScheduler):
    WARMUP_METHODS = ('linear', 'constant')
    # constant -> MultiStepScheduler
    # linear -> with linear warmup
    """
    WarmupStep Learning Rate Scheduler
    With linear warmup schedule and LR Warmup strategy
    """
    def __init__(self, optimizer, milestones: List[int], gamma=0.1, warmup_method: str = 'linear',
                 base_factor=1.0 / 3, warmup_epoch=100, last_epoch=-1):
        assert list(milestones) == sorted(milestones), "LR adjust stages not in ascending order: {}".format(milestones)
        assert warmup_method in self.WARMUP_METHODS, "Warmup method {} not supported!".format(warmup_method)
        self.steps = milestones
        self.gamma = gamma
        self.warmup_method = warmup_method
        self.base_factor = base_factor
        self.warmup_epoch = warmup_epoch
        super(WarmupStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_epoch:  # if still in warmup range
            if self.warmup_method == 'constant':
                warmup_factor = self.warmup_factor
            elif self.warmup_method == 'linear':
                alpha = self.last_epoch / self.warmup_epoch
                warmup_factor = self.base_factor * (1 - alpha) + alpha
            else:
                raise NotImplementedError
        return [base_lr * warmup_factor * self.gamma ** bisect_right(self.steps, self.last_epoch)
                for base_lr in self.base_lrs]


if __name__ == '__main__':
    import torch.optim as optim
    import torch.nn as nn

    net = nn.Linear(10, 10)
    optimizer = optim.Adam(net.parameters(), lr=3.5e-4)
    # scheduler = WarmupMultiStepLR(optimizer, [10, 20], warmup_factor=0.5, warmup_iters=20)
    scheduler = WarmupStepLR(optimizer, [20], base_factor=1/3, warmup_epoch=10)
    for epoch in range(1, 31):  # epoch test
        # for j in range(3):  # iter test
        print("[%d, %.10f]," % (epoch, scheduler.get_lr()[0]), end='')
        optimizer.step()
        scheduler.step()
