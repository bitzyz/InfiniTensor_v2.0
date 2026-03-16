from __future__ import annotations

import tempfile
import importlib
import subprocess
import sys
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union, cast

import torch


class ModelFormat(str, Enum):
    PYTORCH = "pytorch"
    ONNX = "onnx"
    TENSORFLOW = "tensorflow"
    PADDLE = "paddle"


_SUFFIX_TO_FORMAT: Dict[str, ModelFormat] = {
    ".pt": ModelFormat.PYTORCH,
    ".pth": ModelFormat.PYTORCH,
    ".torchscript": ModelFormat.PYTORCH,
    ".ts": ModelFormat.PYTORCH,
    ".jit": ModelFormat.PYTORCH,
    ".onnx": ModelFormat.ONNX,
    ".pb": ModelFormat.TENSORFLOW,
    ".pdmodel": ModelFormat.PADDLE,
}


def _normalize_format(
    format_hint: Optional[Union[str, ModelFormat]],
) -> Optional[ModelFormat]:
    if format_hint is None:
        return None
    if isinstance(format_hint, ModelFormat):
        return format_hint

    normalized = format_hint.strip().lower()
    aliases: Dict[str, ModelFormat] = {
        "pt": ModelFormat.PYTORCH,
        "torch": ModelFormat.PYTORCH,
        "pytorch": ModelFormat.PYTORCH,
        "onnx": ModelFormat.ONNX,
        "tf": ModelFormat.TENSORFLOW,
        "tensorflow": ModelFormat.TENSORFLOW,
        "paddle": ModelFormat.PADDLE,
        "paddlepaddle": ModelFormat.PADDLE,
    }
    if normalized not in aliases:
        raise ValueError(f"Unsupported model format hint: {format_hint}")
    return aliases[normalized]


def detect_model_format(
    model_path: Union[str, Path],
    format_hint: Optional[Union[str, ModelFormat]] = None,
) -> ModelFormat:
    hinted_format = _normalize_format(format_hint)
    if hinted_format is not None:
        return hinted_format

    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model path does not exist: {path}")

    if path.is_dir():
        if (path / "saved_model.pb").exists():
            return ModelFormat.TENSORFLOW

        if (
            (path / "__model__").exists()
            or any(path.glob("*.pdmodel"))
            or (path / "inference.json").exists()
        ):
            return ModelFormat.PADDLE

        raise ValueError(
            f"Unable to infer model format from directory: {path}. "
            "Use format_hint to specify one explicitly."
        )

    suffix = path.suffix.lower()
    if suffix in _SUFFIX_TO_FORMAT:
        return _SUFFIX_TO_FORMAT[suffix]

    if path.name in {"__model__", "inference.json"}:
        return ModelFormat.PADDLE

    raise ValueError(
        f"Unable to infer model format from file: {path}. "
        "Use format_hint to specify one explicitly."
    )


def _load_onnx_as_torch(model_path: Path) -> torch.nn.Module:
    try:
        onnx = importlib.import_module("onnx")
    except ImportError as exc:
        raise ImportError("Loading ONNX models requires dependency 'onnx'.") from exc

    try:
        onnx2torch = importlib.import_module("onnx2torch")
    except ImportError as exc:
        raise ImportError(
            "Loading ONNX models as torch modules requires dependency 'onnx2torch'."
        ) from exc

    onnx_model = onnx.load(model_path.as_posix())
    torch_model = onnx2torch.convert(onnx_model)
    torch_model.eval()
    return torch_model


def _tensorflow_to_onnx_model_path(model_path: Path, opset: int) -> Path:
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp_file:
        onnx_path = Path(tmp_file.name)

    try:
        if model_path.is_dir():
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "tf2onnx.convert",
                    "--saved-model",
                    model_path.as_posix(),
                    "--output",
                    onnx_path.as_posix(),
                    "--opset",
                    str(opset),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    "Failed to convert TensorFlow SavedModel to ONNX: "
                    f"{result.stderr.strip() or result.stdout.strip()}"
                )
        else:
            try:
                tf2onnx = importlib.import_module("tf2onnx")
            except ImportError as exc:
                raise ImportError(
                    "Loading TensorFlow models requires dependency 'tf2onnx'."
                ) from exc

            try:
                tf = importlib.import_module("tensorflow")
            except ImportError as exc:
                raise ImportError(
                    "Converting TensorFlow .pb graph requires dependency 'tensorflow'."
                ) from exc

            graph_def = cast(Any, tf).compat.v1.GraphDef()
            graph_def.ParseFromString(model_path.read_bytes())
            cast(Any, tf2onnx).convert.from_graph_def(
                graph_def,
                name=model_path.stem,
                output_path=onnx_path.as_posix(),
                opset=opset,
            )
    except Exception:
        onnx_path.unlink(missing_ok=True)
        raise

    return onnx_path


def _paddle_to_onnx_model_path(model_path: Path, opset: int) -> Path:
    try:
        paddle2onnx = importlib.import_module("paddle2onnx")
    except ImportError as exc:
        raise ImportError(
            "Loading Paddle models requires dependency 'paddle2onnx'."
        ) from exc

    if model_path.is_dir():
        model_candidates = list(model_path.glob("*.pdmodel"))
        if not model_candidates and (model_path / "inference.json").exists():
            model_candidates = [model_path / "inference.json"]
        params_candidates = list(model_path.glob("*.pdiparams"))
        if not model_candidates or not params_candidates:
            raise ValueError(
                "Paddle directory must contain model file (*.pdmodel or inference.json) "
                "and *.pdiparams file."
            )
        model_file = model_candidates[0]
        params_file = params_candidates[0]
    elif model_path.suffix.lower() in {".pdmodel", ".json"}:
        model_file = model_path
        params_file = model_path.with_suffix(".pdiparams")
        if not params_file.exists():
            raise FileNotFoundError(
                f"Expected Paddle parameter file next to model: {params_file}"
            )
    else:
        raise ValueError(
            "Paddle model path must be a directory with model/params files "
            "or a model file (*.pdmodel or *.json)."
        )

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp_file:
        onnx_path = Path(tmp_file.name)

    try:
        cast(Any, paddle2onnx).export(
            model_filename=model_file.as_posix(),
            params_filename=params_file.as_posix(),
            save_file=onnx_path.as_posix(),
            opset_version=opset,
            enable_onnx_checker=True,
        )
    except Exception:
        onnx_path.unlink(missing_ok=True)
        raise

    return onnx_path


def _load_tensorflow_as_torch(model_path: Path, opset: int) -> torch.nn.Module:
    onnx_path = _tensorflow_to_onnx_model_path(model_path, opset=opset)
    try:
        return _load_onnx_as_torch(onnx_path)
    finally:
        onnx_path.unlink(missing_ok=True)


def _load_paddle_as_torch(model_path: Path, opset: int) -> torch.nn.Module:
    onnx_path = _paddle_to_onnx_model_path(model_path, opset=opset)
    try:
        return _load_onnx_as_torch(onnx_path)
    finally:
        onnx_path.unlink(missing_ok=True)


FrontendLoader = Callable[[Path], torch.nn.Module]


class FrontendModelImporter:
    def __init__(self):
        self._loaders: Dict[ModelFormat, FrontendLoader] = {
            ModelFormat.ONNX: _load_onnx_as_torch,
            ModelFormat.TENSORFLOW: self._tensorflow_loader,
            ModelFormat.PADDLE: self._paddle_loader,
        }
        self._default_opset = 17

    def _tensorflow_loader(self, model_path: Path) -> torch.nn.Module:
        return _load_tensorflow_as_torch(model_path, opset=self._default_opset)

    def _paddle_loader(self, model_path: Path) -> torch.nn.Module:
        return _load_paddle_as_torch(model_path, opset=self._default_opset)

    def set_default_opset(self, opset: int) -> None:
        if opset < 7:
            raise ValueError("ONNX opset must be >= 7")
        self._default_opset = opset

    def register_loader(
        self, model_format: ModelFormat, loader: FrontendLoader
    ) -> None:
        self._loaders[model_format] = loader

    def list_supported_formats(self):
        return sorted(fmt.value for fmt in self._loaders)

    def load(
        self,
        model_path: Union[str, Path],
        format_hint: Optional[Union[str, ModelFormat]] = None,
    ) -> torch.nn.Module:
        path = Path(model_path)
        model_format = detect_model_format(path, format_hint=format_hint)
        if model_format == ModelFormat.PYTORCH:
            raise ValueError(
                "PyTorch format should be handled directly by passing a torch.nn.Module "
                "into TorchFXTranslator.import_from_fx()."
            )

        if model_format not in self._loaders:
            raise ValueError(
                f"No loader registered for model format: {model_format.value}"
            )

        return self._loaders[model_format](path)


default_frontend_importer = FrontendModelImporter()


def load_model_as_torch(
    model_path: Union[str, Path],
    format_hint: Optional[Union[str, ModelFormat]] = None,
) -> torch.nn.Module:
    return default_frontend_importer.load(model_path, format_hint=format_hint)
