from abc import ABC, abstractmethod


class AbsContrastiveModel(ABC):
    @abstractmethod
    def encode(self, **kwargs):
        pass

    @abstractmethod
    def forward(self, **kwargs):
        pass
