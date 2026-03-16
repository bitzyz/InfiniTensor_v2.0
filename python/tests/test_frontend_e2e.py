import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
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


def test_onnx_frontend_e2e():
    class Toy(torch.nn.Module):
        def forward(self, x):
            return x * 2.0 + 1.0

    model = Toy().eval()
    x = torch.randn(2, 3, dtype=torch.float32)

    with tempfile.TemporaryDirectory() as td:
        onnx_path = Path(td) / "toy.onnx"
        torch.onnx.export(
            model,
            (x,),
            onnx_path.as_posix(),
            input_names=["x"],
            output_names=["y"],
            opset_version=17,
        )

        loaded = frontend.load_model_as_torch(onnx_path)
        loaded.eval()

        y_ref = model(x)
        y_loaded = loaded(x)
        max_diff = (y_ref - y_loaded).abs().max().item()
        assert max_diff < 1e-5


def test_tensorflow_savedmodel_frontend_e2e():
    tf = __import__("tensorflow")

    class ToyTF(tf.Module):
        @tf.function(input_signature=[tf.TensorSpec(shape=[None, 3], dtype=tf.float32)])
        def __call__(self, x):
            return {"y": x * 3.0 - 2.0}

    x_np = np.random.randn(4, 3).astype(np.float32)
    x_torch = torch.from_numpy(x_np)

    with tempfile.TemporaryDirectory() as td:
        saved_dir = Path(td) / "saved_model"
        tf.saved_model.save(ToyTF(), saved_dir.as_posix())

        loaded = frontend.load_model_as_torch(saved_dir)
        loaded.eval()

        y_ref = x_torch * 3.0 - 2.0
        y_loaded = loaded(x_torch)
        max_diff = (y_ref - y_loaded).abs().max().item()
        assert max_diff < 1e-4


def test_paddle_frontend_e2e():
    script = """
import importlib.util
import tempfile
from pathlib import Path
import numpy as np
import torch
import paddle

module_path = Path('/home/luoyue/compiler/InfiniTensor_v2.0/python/src/infinitensor/model_frontend.py')
spec = importlib.util.spec_from_file_location('model_frontend', module_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

paddle.enable_static()
place = paddle.CPUPlace()
exe = paddle.static.Executor(place)
startup = paddle.static.Program()
main = paddle.static.Program()

with paddle.static.program_guard(main, startup):
    x = paddle.static.data(name='x', shape=[-1, 3], dtype='float32')
    w = paddle.create_parameter(
        shape=[1],
        dtype='float32',
        default_initializer=paddle.nn.initializer.Constant(0.5),
    )
    y = x * 4.0 + w

exe.run(startup)
x_np = np.random.randn(5, 3).astype('float32')
x_torch = torch.from_numpy(x_np)

with tempfile.TemporaryDirectory() as td:
    save_dir = Path(td) / 'paddle_model'
    save_dir.mkdir(parents=True, exist_ok=True)
    paddle.static.save_inference_model(
        path_prefix=(save_dir / 'inference').as_posix(),
        feed_vars=[x],
        fetch_vars=[y],
        executor=exe,
        program=main,
    )
    loaded = module.load_model_as_torch(save_dir)
    loaded.eval()

    y_ref = x_torch * 4.0 + 0.5
    y_loaded = loaded(x_torch)
    max_diff = (y_ref - y_loaded).abs().max().item()
    assert max_diff < 1e-4
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr or result.stdout
