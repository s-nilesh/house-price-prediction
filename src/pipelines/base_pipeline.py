from abc import ABC, abstractmethod

class BasePipeline(ABC):
    """
    Abstract class for defining a pipeline step (feature_eng, data_load, training, inference, etc.)
    """

    @abstractmethod
    def execute(self):
        """
        Execute the pipeline step.
        """
        pass