import torch
from abc import abstractmethod
from numpy import inf

class BaseTrainer:
    """
    Base class for trainers
    """
    @abstractmethod
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch
        :param epoch: Current epoch number
        """
        raise NotImplementedError

    @abstractmethod
    def train(self):
        """
        Training logic for an epoch
        :param epoch: Current epoch number
        """
        raise NotImplementedError

