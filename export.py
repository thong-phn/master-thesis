from pathlib import Path

import torch


def export_best_model_to_onnx(
    model_class,
    checkpoint_path,
    onnx_path,
    num_channels,
    input_len,
    device=None,
    opset_version=17,
):
    """
    Load a trained checkpoint and export the model to ONNX.

    Args:
        model_class: Model class to instantiate.
        checkpoint_path: Path to .pth checkpoint.
        onnx_path: Destination ONNX file path.
        num_channels: Number of input channels (3 accel / 6 accel+gyro).
        input_len: Input sequence length (e.g., 65 for FFT, 128 for raw).
        device: torch.device, optional.
        opset_version: ONNX opset version.

    Returns:
        str: exported ONNX file path.
    """
    checkpoint_path = Path(checkpoint_path)
    onnx_path = Path(onnx_path)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model_class(num_channels=num_channels, freq_bins=input_len).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # batch: 1, channel: 6, length: 65
    dummy_input = torch.randn(1, num_channels, input_len, device=device)

    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        dynamo=False,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        verbose=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
    )

    return str(onnx_path)
