"""Typed inference requests and results."""

from pydantic import BaseModel, Field, model_validator


class Example(BaseModel):
    """A text with its known answer, supplied to demonstrate a query's task.

    Args:
        state: Example text, separate from the text being classified.
        choice: Correct answer for this example. Must exactly match one
            of the containing query's options.

    Examples:
        >>> example = Example(
        ...     state="We will recruit more teachers.",
        ...     choice="Education",
        ... )
    """

    state: str
    choice: str


class Query(BaseModel):
    """A question answered by selecting one of the supplied options.

    Args:
        text: Question to ask about the request's text.
        options: Nonempty list of distinct answer strings. The model returns
            one of these strings and a probability for every option.
        examples: Optional texts with known answers demonstrating this
            question. Each example's choice must appear in options.

    Examples:
        >>> query = Query(
        ...     text="Which policy area does this statement concern?",
        ...     options=["Education", "Transport"],
        ...     examples=[
        ...         Example(
        ...             state="We will recruit more teachers.",
        ...             choice="Education",
        ...         )
        ...     ],
        ... )
    """

    text: str
    options: list[str] = Field(min_length=1)
    examples: list[Example] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_options(self):
        assert len(self.options) == len(set(self.options))
        assert all(example.choice in self.options for example in self.examples)
        return self


class Request(BaseModel):
    """A text and the questions to answer about it.

    Args:
        state: Complete text made available to every query in this request.
        queries: Nonempty list of classification questions. Each question
            receives its own result, in the supplied order.

    Examples:
        >>> request = Request(
        ...     state="We will expand the bus network.",
        ...     queries=[
        ...         Query(
        ...             text="Which policy area does this statement concern?",
        ...             options=["Education", "Transport"],
        ...         )
        ...     ],
        ... )
    """

    state: str
    queries: list[Query] = Field(min_length=1)


class Result(BaseModel):
    choice: str
    probabilities: dict[str, float]
