class SpeechProcessingError(Exception):
    """Base exception for the project."""


class AudioProcessingError(SpeechProcessingError):
    """Raised when audio inference fails."""


class JSONParseError(SpeechProcessingError):
    """Raised when judge output cannot be parsed."""
