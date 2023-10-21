import torch
from pathlib import Path
from typing import List, Any, Dict, Union
from marshmallow.fields import Field
from dataclasses import asdict
from marshmallow_dataclass import dataclass, NewType
from robot_utils.serialize.dataclass import load_dataclass, dump_data_to_yaml, default_field


@dataclass
class _TorchTensorDTO:
    dtype:  str
    data:   List[Any]


class TorchField(Field):
    def __init__(
            self, *args, **kwargs
    ):
        super(TorchField, self).__init__(*args, **kwargs)

    def _serialize(self, value: torch.Tensor, *args, **kwargs):
        if value is None:
            return None

        return asdict(_TorchTensorDTO(dtype=str(value.dtype).split(".")[1], data=value.numpy().tolist()))

    def _deserialize(self, value, *args, **kwargs):
        if value is None:
            return None

        torch_tensor_obj = _TorchTensorDTO(**value)
        return torch.tensor(torch_tensor_obj.data, dtype=getattr(torch, torch_tensor_obj.dtype))


TorchTensor = NewType("NdArray", torch.tensor, field=TorchField)


@dataclass
class DictTorchTensor:
    data: Dict[str, TorchTensor] = default_field({})

    @classmethod
    def from_yaml(cls, filename: Union[str, Path]):
        return load_dataclass(DictTorchTensor, filename)

    @classmethod
    def to_yaml(cls, data, filename: Union[str, Path]):
        dump_data_to_yaml(DictTorchTensor, data, filename)


__all__ = [TorchField, TorchTensor]
