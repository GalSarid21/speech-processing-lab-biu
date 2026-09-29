from pydantic import BaseModel, Field


class ItemMetadata(BaseModel):
    """Typed per-item context carried from dataset loading through to scoring."""

    item_id: str
    question: str
    choices: list[str] = Field(description="Choices in the order SHOWN to the model.")
    transcript: str | None = None
    modality: str = ""
    category: str = ""
    sub_category: str = Field(default="", description='MMAR column "sub-category".')

    original_item_id: str | None = None
    variant_idx: int | None = None
    permutation: list[int] | None = Field(default=None, description="permutation[displayed_index] = original_index")
    original_choices: list[str] | None = None
    votes: list[str] | None = None

    choice_scores: dict[str, float] | None = Field(default=None, description="T3 diagnostics.")
    cycle_spans: list[tuple[float, float]] | None = Field(
        default=None, description="ICBHI breathing-cycle boundaries, in seconds."
    )
