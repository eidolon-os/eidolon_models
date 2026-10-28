"""Export the pinned native CAMPPlus checkpoint; never edits a public ONNX graph."""
import argparse
import hashlib
import json
from pathlib import Path

import onnx
import torch
from modelscope.models.audio.sv.DTDNN import CAMPPlus
from torch.onnx import symbolic_helper


@symbolic_helper.parse_args("v", "is", "is", "is", "i", "i")
def camp_segment_average(g, x, kernel, stride, padding, ceil_mode, count_include_pad):
    """Translate this model's unpadded ceil average with its actual Torch denominator.

    Torch divides a final partial segment by its valid length. ONNX's
    count_include_pad=1 includes the implicit tail. Restrict this translation
    to the audited CAM segment operator; fail closed on any other configuration.
    """
    if (kernel, stride, padding, ceil_mode, count_include_pad) != ([100], [100], [0], 1, 1):
        raise ValueError("Unaudited average-pooling configuration")
    return g.op("AveragePool", x, kernel_shape_i=kernel, strides_i=stride,
                pads_i=[0, 0], ceil_mode_i=1, count_include_pad_i=0)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("weights", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--exporter", choices=["legacy", "dynamo", "semantic"], default="semantic")
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    model = CAMPPlus(feat_dim=80, embedding_size=192).eval()
    model.load_state_dict(torch.load(args.weights, map_location="cpu", weights_only=True), strict=True)
    example = torch.randn(1, 398, 80)
    kwargs = dict(input_names=["features"], output_names=["embedding"],
                  external_data=False, opset_version=18)
    if args.exporter == "dynamo":
        kwargs.update(dynamo=True, dynamic_shapes=({1: torch.export.Dim("frames", min=30, max=3000)},),
                      report=True, verify=True, artifacts_dir=str(args.output.parent))
    else:
        kwargs.update(dynamo=False, dynamic_axes={"features": {1: "frames"}})
        if args.exporter == "semantic":
            torch.onnx.register_custom_op_symbolic("aten::avg_pool1d", camp_segment_average, 18)
    with torch.no_grad():
        torch.onnx.export(model, (example,), str(args.output), **kwargs)
    graph = onnx.load(args.output)
    onnx.checker.check_model(graph, full_check=True)
    if args.exporter == "semantic":
        pools = [n for n in graph.graph.node if n.op_type == "AveragePool"]
        assert len(pools) == 52
        assert all(next(a.i for a in n.attribute if a.name == "count_include_pad") == 0 for n in pools)
    result = dict(source_weights_sha256=sha(args.weights), onnx_sha256=sha(args.output),
                  bytes=args.output.stat().st_size, torch=torch.__version__, onnx=onnx.__version__,
                  exporter=args.exporter, opsets=[dict(domain=x.domain, version=x.version) for x in graph.opset_import],
                  input_contract="float32 [1,T,80] mean-centered FBank, batch=1, 30<=T<=3000",
                  output_contract="float32 [1,192], not L2 normalized",
                  pools=[{a.name: onnx.helper.get_attribute_value(a) for a in n.attribute}
                         for n in graph.graph.node if n.op_type == "AveragePool"])
    args.output.with_suffix(".json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "pools"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
