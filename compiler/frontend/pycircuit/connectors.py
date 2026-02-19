from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Iterator, Mapping, MutableMapping


class ConnectorError(TypeError):
    pass


class Connector:
    """Base class for inter-module connection objects."""

    owner: Any
    name: str

    @property
    def ty(self) -> str:
        raise NotImplementedError

    def read(self) -> Any:
        raise NotImplementedError

    def out(self) -> Any:
        """Read connector payload as a value suitable for expressions."""
        value = self.read()
        if hasattr(value, "out"):
            return value.out()
        return value

    def __bool__(self) -> bool:
        raise TypeError(
            "Connector cannot be used as a Python boolean. "
            "Use hardware comparisons/selects and keep conditions as i1 values."
        )

    @property
    def width(self) -> int:
        return int(getattr(self.out(), "width"))

    def __getitem__(self, idx: Any) -> Any:
        return self.out()[idx]

    def __getattr__(self, name: str) -> Any:
        return getattr(self.out(), name)

    def __add__(self, other: Any) -> Any:
        return self.out() + other

    def __radd__(self, other: Any) -> Any:
        return other + self.out()

    def __sub__(self, other: Any) -> Any:
        return self.out() - other

    def __rsub__(self, other: Any) -> Any:
        return other - self.out()

    def __mul__(self, other: Any) -> Any:
        return self.out() * other

    def __rmul__(self, other: Any) -> Any:
        return other * self.out()

    def __floordiv__(self, other: Any) -> Any:
        return self.out() // other

    def __rfloordiv__(self, other: Any) -> Any:
        return other // self.out()

    def __truediv__(self, other: Any) -> Any:
        return self.out() / other

    def __rtruediv__(self, other: Any) -> Any:
        return other / self.out()

    def __mod__(self, other: Any) -> Any:
        return self.out() % other

    def __rmod__(self, other: Any) -> Any:
        return other % self.out()

    def __and__(self, other: Any) -> Any:
        return self.out() & other

    def __rand__(self, other: Any) -> Any:
        return other & self.out()

    def __or__(self, other: Any) -> Any:
        return self.out() | other

    def __ror__(self, other: Any) -> Any:
        return other | self.out()

    def __xor__(self, other: Any) -> Any:
        return self.out() ^ other

    def __rxor__(self, other: Any) -> Any:
        return other ^ self.out()

    def __invert__(self) -> Any:
        return ~self.out()

    def __lshift__(self, other: Any) -> Any:
        return self.out() << other

    def __rshift__(self, other: Any) -> Any:
        return self.out() >> other

    def __eq__(self, other: object) -> Any:  # type: ignore[override]
        return self.out() == other

    def __ne__(self, other: object) -> Any:  # type: ignore[override]
        return self.out() != other

    def __lt__(self, other: Any) -> Any:
        return self.out() < other

    def __gt__(self, other: Any) -> Any:
        return self.out() > other

    def __le__(self, other: Any) -> Any:
        return self.out() <= other

    def __ge__(self, other: Any) -> Any:
        return self.out() >= other


@dataclass(frozen=True, eq=False)
class WireConnector(Connector):
    owner: Any
    name: str
    wire: Any

    def __post_init__(self) -> None:
        maybe_owner = getattr(self.wire, "m", None)
        if maybe_owner is not None and maybe_owner is not self.owner:
            raise ConnectorError("wire connector must belong to the declaring Circuit")
        if not hasattr(self.wire, "ty"):
            raise ConnectorError("wire connector payload must expose a `ty` attribute")

    @property
    def ty(self) -> str:
        return str(self.wire.ty)

    @property
    def signed(self) -> bool:
        return bool(getattr(self.wire, "signed", False))

    def read(self) -> Any:
        return self.wire


@dataclass(frozen=True, eq=False)
class RegConnector(Connector):
    owner: Any
    name: str
    reg: Any

    def __post_init__(self) -> None:
        q = getattr(self.reg, "q", None)
        if q is None or getattr(q, "m", None) is not self.owner:
            raise ConnectorError("reg connector must belong to the declaring Circuit")

    @property
    def ty(self) -> str:
        return str(self.reg.ty)

    def read(self) -> Any:
        return self.reg.q

    def set(self, value: Any, *, when: Any = 1) -> None:
        self.reg.set(value, when=when)


class ConnectorBundle:
    def __init__(self, fields: Mapping[str, Connector]) -> None:
        out: MutableMapping[str, Connector] = {}
        for k, v in fields.items():
            key = str(k)
            if not key:
                raise ConnectorError("bundle field name must be non-empty")
            if not isinstance(v, Connector):
                raise ConnectorError(f"bundle field {key!r}: expected Connector, got {type(v).__name__}")
            out[key] = v
        self.fields: dict[str, Connector] = dict(out)

    def __getitem__(self, key: str) -> Connector:
        return self.fields[str(key)]

    def __iter__(self) -> Iterator[str]:
        return iter(self.fields)

    def __len__(self) -> int:
        return len(self.fields)

    def items(self) -> Iterable[tuple[str, Connector]]:
        return self.fields.items()

    def values(self) -> Iterable[Connector]:
        return self.fields.values()

    def keys(self) -> Iterable[str]:
        return self.fields.keys()


@dataclass(frozen=True)
class ModuleInstanceHandle:
    name: str
    symbol: str
    inputs: Mapping[str, Connector]
    outputs: Connector | ConnectorBundle


ConnectorLike = Connector | ConnectorBundle


def is_connector(v: Any) -> bool:
    return isinstance(v, Connector)


def is_connector_bundle(v: Any) -> bool:
    return isinstance(v, ConnectorBundle)


def connector_owner(v: ConnectorLike) -> Any | None:
    if isinstance(v, Connector):
        return v.owner
    if isinstance(v, ConnectorBundle):
        owner: Any | None = None
        for c in v.values():
            if owner is None:
                owner = c.owner
                continue
            if c.owner is not owner:
                raise ConnectorError("connector bundle fields must belong to the same Circuit")
        return owner
    return None


def connector_to_wire(v: Connector, *, ctx: str) -> Any:
    if not isinstance(v, Connector):
        raise ConnectorError(f"{ctx}: expected Connector, got {type(v).__name__}")
    return v.read()
