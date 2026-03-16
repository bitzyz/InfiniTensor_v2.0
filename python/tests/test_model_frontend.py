import importlib.util
from pathlib import Path

import pytest
import torch


def _load_model_frontend_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "infinitensor"
        / "model_frontend.py"
    )
    spec = importlib.util.spec_from_file_location(
        "infinitensor_model_frontend", module_path
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


frontend = _load_model_frontend_module()


def test_detect_model_format_by_file_extension(tmp_path):
    onnx_file = tmp_path / "model.onnx"
    onnx_file.write_bytes(b"onnx")
    assert frontend.detect_model_format(onnx_file) == frontend.ModelFormat.ONNX

    tf_file = tmp_path / "graph.pb"
    tf_file.write_bytes(b"pb")
    assert frontend.detect_model_format(tf_file) == frontend.ModelFormat.TENSORFLOW

    paddle_file = tmp_path / "inference.pdmodel"
    paddle_file.write_bytes(b"pdmodel")
    assert frontend.detect_model_format(paddle_file) == frontend.ModelFormat.PADDLE


def test_detect_model_format_for_directories(tmp_path):
    tf_dir = tmp_path / "saved_model"
    tf_dir.mkdir()
    (tf_dir / "saved_model.pb").write_bytes(b"pb")
    assert frontend.detect_model_format(tf_dir) == frontend.ModelFormat.TENSORFLOW

    paddle_dir = tmp_path / "paddle_model"
    paddle_dir.mkdir()
    (paddle_dir / "inference.pdmodel").write_bytes(b"pdmodel")
    (paddle_dir / "inference.pdiparams").write_bytes(b"params")
    assert frontend.detect_model_format(paddle_dir) == frontend.ModelFormat.PADDLE


def test_detect_model_format_hint_overrides_path(tmp_path):
    unknown_file = tmp_path / "weights.bin"
    unknown_file.write_bytes(b"data")
    fmt = frontend.detect_model_format(unknown_file, format_hint="onnx")
    assert fmt == frontend.ModelFormat.ONNX


def test_detect_model_format_unknown_raises(tmp_path):
    unknown_file = tmp_path / "weights.bin"
    unknown_file.write_bytes(b"data")
    with pytest.raises(ValueError):
        frontend.detect_model_format(unknown_file)


def test_frontend_importer_custom_loader_dispatch(tmp_path):
    class DummyModule(torch.nn.Module):
        def forward(self, x):
            return x

    onnx_file = tmp_path / "model.onnx"
    onnx_file.write_bytes(b"onnx")

    importer = frontend.FrontendModelImporter()
    importer.register_loader(frontend.ModelFormat.ONNX, lambda _: DummyModule())
    module = importer.load(onnx_file)
    assert isinstance(module, DummyModule)
