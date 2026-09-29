class SpeechProcessingError(Exception):
    """Base exception for the project."""


class AudioProcessingError(SpeechProcessingError):
    """Raised when audio inference fails."""


class JSONParseError(SpeechProcessingError):
    """Raised when judge output cannot be parsed."""


class DatasetIntegrityError(SpeechProcessingError):
    """Raised when dataset rows or derived artifacts violate the expected contract."""


class FewShotLeakageError(SpeechProcessingError):
    """Raised when a few-shot example overlaps the evaluation set."""


class DeprecatedExperimentError(SpeechProcessingError):
    """Raised when a retired experiment version is requested."""


class ContrastiveScoringError(SpeechProcessingError):
    """Raised when contrastive choice scoring cannot produce a trustworthy score."""


class InvalidExperimentConfigError(SpeechProcessingError):
    """Raised when an experiment definition combines mutually incompatible options."""
