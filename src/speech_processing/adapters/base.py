import abc
from typing import Any


class BaseGenerationAdapter(abc.ABC):
    @abc.abstractmethod
    def generate_batch(self, *args: Any, **kwargs: Any) -> list[str]:
        """Generate text outputs for a batch of inputs."""
