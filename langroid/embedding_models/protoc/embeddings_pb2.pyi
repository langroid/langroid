from typing import (
    ClassVar as _ClassVar,
)
from typing import (
    Iterable as _Iterable,
)
from typing import (
    Mapping as _Mapping,
)
from typing import (
    Optional as _Optional,
)
from typing import (
    Union as _Union,
)

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf.internal import containers as _containers

DESCRIPTOR: _descriptor.FileDescriptor

class EmbeddingRequest(_message.Message):
    __slots__ = ("model_name", "batch_size", "strings")
    MODEL_NAME_FIELD_NUMBER: _ClassVar[int]
    BATCH_SIZE_FIELD_NUMBER: _ClassVar[int]
    STRINGS_FIELD_NUMBER: _ClassVar[int]
    model_name: str
    batch_size: int
    strings: _containers.RepeatedScalarFieldContainer[str]
    def __init__(
        self,
        model_name: _Optional[str] = ...,
        batch_size: _Optional[int] = ...,
        strings: _Optional[_Iterable[str]] = ...,
    ) -> None: ...

class BatchEmbeds(_message.Message):
    __slots__ = ("embeds",)
    EMBEDS_FIELD_NUMBER: _ClassVar[int]
    embeds: _containers.RepeatedCompositeFieldContainer[Embed]
    def __init__(
        self, embeds: _Optional[_Iterable[_Union[Embed, _Mapping]]] = ...
    ) -> None: ...

class Embed(_message.Message):
    __slots__ = ("embed",)
    EMBED_FIELD_NUMBER: _ClassVar[int]
    embed: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, embed: _Optional[_Iterable[float]] = ...) -> None: ...
