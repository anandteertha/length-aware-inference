import numpy as np
import pytest

from etdl.variants import pad_inputs, weights_digest


def test_padding_preserves_tokens_and_masks_new_positions():
    inputs = {
        "input_ids": np.array([[101, 42, 102]], dtype=np.int64),
        "attention_mask": np.ones((1, 3), dtype=np.int64),
    }
    padded = pad_inputs(inputs, 5, 0)
    np.testing.assert_array_equal(padded["input_ids"], [[101, 42, 102, 0, 0]])
    np.testing.assert_array_equal(padded["attention_mask"], [[1, 1, 1, 0, 0]])
    assert inputs["input_ids"].shape == (1, 3)


def test_fixed_variant_cannot_truncate_overflow():
    inputs = {"input_ids": np.ones((1, 17)), "attention_mask": np.ones((1, 17))}
    with pytest.raises(ValueError, match="no truncation"):
        pad_inputs(inputs, 16, 0)


def test_shape_specialization_preserves_weights_and_executes_two_lengths(tmp_path):
    import onnx
    import onnxruntime as ort
    from onnx import TensorProto, helper
    from onnxruntime.tools.onnx_model_utils import make_input_shape_fixed

    graph = helper.make_graph(
        [helper.make_node("Mul", ["input_ids", "weight"], ["result"])],
        "toy",
        [helper.make_tensor_value_info("input_ids", TensorProto.FLOAT, [1, "sequence"])],
        [helper.make_tensor_value_info("result", TensorProto.FLOAT, [1, "sequence"])],
        [helper.make_tensor("weight", TensorProto.FLOAT, [], [2.0])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=9)
    before = weights_digest(model)
    for length in (16, 32):
        fixed = onnx.ModelProto()
        fixed.CopyFrom(model)
        make_input_shape_fixed(fixed.graph, "input_ids", [1, length])
        assert weights_digest(fixed) == before
        path = tmp_path / f"fixed-{length}.onnx"
        onnx.save(fixed, path)
        runtime = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        output = runtime.run(None, {"input_ids": np.ones((1, length), dtype=np.float32)})[0]
        np.testing.assert_array_equal(output, np.full((1, length), 2.0))
