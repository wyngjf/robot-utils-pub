from pathlib import Path
from marshmallow.fields import Field
from marshmallow_dataclass import NewType, dataclass


class PathField(Field):
    def __init__(
            self, *args, **kwargs
    ):
        super(PathField, self).__init__(*args, **kwargs)

    def _serialize(self, value: Path, *args, **kwargs):
        if value is None:
            return None

        return str(value)

    def _deserialize(self, value, *args, **kwargs):
        if value is None:
            return None

        return Path(value)


PathT = NewType("Path", Path, field=PathField)

__all__ = [PathT, PathField]


if __name__ == "__main__":
    from robot_utils.serialize.dataclass import load_dataclass, dump_data_to_yaml, default_field

    @dataclass
    class Test:
        path: PathT

    d = dict(
        path="./example"
    )

    test = load_dataclass(Test, d)
    ic(type(test.path))
    ic(test.path)
    ic(test.path.exists())
    dump_data_to_yaml(Test, test, "./example/test.yaml")

