import numpy as np
from pathlib import Path
from typing import List, Any, Dict, Union
from marshmallow.fields import Field
from dataclasses import asdict
from marshmallow_dataclass import NewType, dataclass
from robot_utils.serialize.dataclass import load_dataclass, dump_data_to_yaml, default_field


@dataclass
class _NumpyArrayDTO:
    dtype:  str
    data:   List[Any]


class NumpyField(Field):
    def __init__(
            self, *args, **kwargs
    ):
        super(NumpyField, self).__init__(*args, **kwargs)

    def _serialize(self, value: np.ndarray, *args, **kwargs):
        if value is None:
            return None

        return asdict(_NumpyArrayDTO(dtype=value.dtype.name, data=value.tolist()))

    def _deserialize(self, value, *args, **kwargs):
        if value is None:
            return None

        np_array_obj = _NumpyArrayDTO(**value)

        return np.array(np_array_obj.data, dtype=np.dtype(np_array_obj.dtype))


NumpyArray = NewType("NdArray", np.ndarray, field=NumpyField)


@dataclass
class DictNumpyArray:
    data: Dict[str, NumpyArray] = default_field({})

    @classmethod
    def from_yaml(cls, filename: Union[str, Path]):
        return load_dataclass(DictNumpyArray, filename)

    @classmethod
    def to_yaml(cls, data, filename: Union[str, Path]):
        dump_data_to_yaml(DictNumpyArray, data, filename)


@dataclass
class NumpyArrayData:
    data: NumpyArray

    @classmethod
    def from_yaml(cls, filename: Union[str, Path]):
        return load_dataclass(NumpyArrayData, filename).data

    @classmethod
    def to_yaml(cls, data: np.ndarray, filename: Union[str, Path]):
        dump_data_to_yaml(NumpyArrayData, NumpyArrayData(data), filename)


__all__ = [NumpyField, NumpyArray]
