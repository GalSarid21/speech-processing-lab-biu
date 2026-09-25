import abc

from speech_processing.data.dtos import (
    AudioRequest,
    AudioResponse,
    JudgeRequest,
    JudgeResponse,
    TextRequest,
    TextResponse
)


class BaseAudioModel(abc.ABC):
    @abc.abstractmethod
    def batch_infer(self, requests: list[AudioRequest]) -> list[AudioResponse]:
        pass


class BaseTextModel(abc.ABC):
    @abc.abstractmethod
    def batch_infer(self, requests: list[TextRequest]) -> list[TextResponse]:
        pass


class BaseJudgeModel(abc.ABC):
    @abc.abstractmethod
    def batch_evaluate(self, requests: list[JudgeRequest]) -> list[JudgeResponse]:
        pass
