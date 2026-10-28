"""Validate the export translation at every tail length, including tiny segments."""
import importlib.util
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

spec = importlib.util.spec_from_file_location("camp_export", Path(__file__).with_name("export.py"))
export = importlib.util.module_from_spec(spec)
spec.loader.exec_module(export)


def test_segment_average_export_all_remainders(tmp_path):
    class Segment(torch.nn.Module):
        def forward(self, x):
            return torch.nn.functional.avg_pool1d(x, 100, 100, ceil_mode=True)

    torch.set_num_threads(1)
    torch.manual_seed(42)
    module = Segment().eval()
    path = tmp_path / "segment.onnx"
    torch.onnx.register_custom_op_symbolic("aten::avg_pool1d", export.camp_segment_average, 18)
    try:
        torch.onnx.export(module, (torch.ones(1, 3, 201),), str(path), dynamo=False,
                          opset_version=18, input_names=["x"], dynamic_axes={"x": {2: "time"}})
    finally:
        torch.onnx.unregister_custom_op_symbolic("aten::avg_pool1d", 18)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
    for length in list(range(1, 402)) + [999, 1000, 1001, 2999, 3000]:
        for x in [torch.ones(1, 3, length), torch.randn(1, 3, length)]:
            expected = module(x).numpy()
            actual = session.run(None, {"x": x.numpy()})[0]
            np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
