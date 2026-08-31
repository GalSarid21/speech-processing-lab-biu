import abc

from speech_processing.data.dtos import JudgeRequest


class BasePromptTemplate(abc.ABC):
    @abc.abstractmethod
    def build_conversation(self, req: JudgeRequest) -> list[dict[str, str]]:
        pass
