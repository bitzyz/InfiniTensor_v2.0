"""
Unit tests for pyinfinitensor DType API.

Tests dtype_from_string() / dtype_to_string() round-trips and the DType
class properties (size, name, equality operators).
"""

import pytest
import pyinfinitensor as pit


# ---------------------------------------------------------------------------
# dtype_from_string → dtype_to_string round-trip
# ---------------------------------------------------------------------------

# (input alias, expected canonical name returned by dtype_to_string)
ROUNDTRIP_CASES = [
    ("float32",       "float32"),
    ("torch.float32", "float32"),
    ("torch.float",   "float32"),
    ("float16",       "float16"),
    ("torch.float16", "float16"),
    ("torch.half",    "float16"),
    ("float64",       "float64"),
    ("double",        "float64"),
    ("torch.double",  "float64"),
    ("int8",          "int8"),
    ("torch.int8",    "int8"),
    ("int16",         "int16"),
    ("torch.int16",   "int16"),
    ("int32",         "int32"),
    ("torch.int32",   "int32"),
    ("torch.int",     "int32"),
    ("int64",         "int64"),
    ("torch.int64",   "int64"),
    ("torch.long",    "int64"),
    ("uint8",         "uint8"),
    ("torch.uint8",   "uint8"),
    ("bool",          "bool"),
    ("torch.bool",    "bool"),
    ("bfloat16",      "bfloat16"),
]


@pytest.mark.parametrize("alias,canonical", ROUNDTRIP_CASES)
def test_from_string_roundtrip(alias, canonical):
    dt = pit.dtype_from_string(alias)
    result = pit.dtype_to_string(dt)
    assert result == canonical, f"alias '{alias}': expected '{canonical}', got '{result}'"


# ---------------------------------------------------------------------------
# Unknown string raises an exception
# ---------------------------------------------------------------------------

def test_from_string_unknown_raises():
    with pytest.raises(Exception, match="Unknown"):
        pit.dtype_from_string("not_a_dtype")


# ---------------------------------------------------------------------------
# DType class properties
# ---------------------------------------------------------------------------

def test_dtype_name_property():
    dt = pit.dtype_from_string("float32")
    # DataType.toString() returns the abbreviated uppercase name used internally
    assert dt.name == "F32"


def test_dtype_size_float32():
    dt = pit.dtype_from_string("float32")
    assert dt.size == 4


def test_dtype_size_float16():
    dt = pit.dtype_from_string("float16")
    assert dt.size == 2


def test_dtype_size_float64():
    dt = pit.dtype_from_string("float64")
    assert dt.size == 8


def test_dtype_size_int8():
    dt = pit.dtype_from_string("int8")
    assert dt.size == 1


def test_dtype_size_int32():
    dt = pit.dtype_from_string("int32")
    assert dt.size == 4


def test_dtype_size_int64():
    dt = pit.dtype_from_string("int64")
    assert dt.size == 8


# ---------------------------------------------------------------------------
# Equality / inequality
# ---------------------------------------------------------------------------

def test_dtype_equality():
    a = pit.dtype_from_string("float32")
    b = pit.dtype_from_string("torch.float32")
    assert a == b


def test_dtype_inequality():
    a = pit.dtype_from_string("float32")
    b = pit.dtype_from_string("float16")
    assert a != b


# ---------------------------------------------------------------------------
# DType type identity checks (via equality / to_string)
# ---------------------------------------------------------------------------

def test_dtype_get_type():
    # get_type() returns the underlying C enum (infiniDtype_t), which is
    # not registered in Python so cannot be compared directly. Verify
    # consistency via equality operators and to_string() instead.
    dt = pit.dtype_from_string("float32")
    dt_alias = pit.dtype_from_string("torch.float")  # same underlying type
    assert dt == dt_alias
    assert dt.to_string() == dt_alias.to_string()
    # Different dtype must not be equal
    dt_f16 = pit.dtype_from_string("float16")
    assert dt != dt_f16
