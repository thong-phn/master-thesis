import argparse
from pathlib import Path

import numpy as np

from train import MyDataset


def parse_subject_ids(value):
    if value is None or value.strip() == "":
        return None
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def select_indices(labels, sample_count, seed=42, sampling_method="random"):
    rng = np.random.default_rng(seed)
    total = len(labels)
    sample_count = min(sample_count, total)

    if sampling_method == "random":
        return np.sort(rng.choice(total, size=sample_count, replace=False))

    if sampling_method != "stratified":
        raise ValueError("sampling_method must be 'random' or 'stratified'.")

    class_ids = np.unique(labels)
    n_classes = len(class_ids)
    if n_classes == 0:
        raise ValueError("No class labels found for stratified sampling.")

    per_class = sample_count // n_classes
    selected_parts = []
    remainder_pools = []

    for class_id in class_ids:
        class_indices = np.where(labels == class_id)[0]
        shuffled = rng.permutation(class_indices)
        take = min(per_class, len(shuffled))
        selected_parts.append(shuffled[:take])
        if take < len(shuffled):
            remainder_pools.append(shuffled[take:])

    selected = np.concatenate(selected_parts) if selected_parts else np.array([], dtype=np.int64)
    remaining = sample_count - len(selected)

    if remaining > 0:
        if remainder_pools:
            remainder_pool = np.concatenate(remainder_pools)
        else:
            remainder_pool = np.array([], dtype=np.int64)

        if len(remainder_pool) > 0:
            extra_take = min(remaining, len(remainder_pool))
            extra = rng.choice(remainder_pool, size=extra_take, replace=False)
            selected = np.concatenate([selected, extra])

    return np.sort(selected.astype(np.int64))


def get_onnx_input_spec(onnx_path):
    try:
        import onnx
    except Exception as exc:
        raise RuntimeError("onnx is required when --onnx-path is provided.") from exc

    model = onnx.load(str(onnx_path))
    input_tensor = model.graph.input[0]
    input_name = input_tensor.name
    dims = input_tensor.type.tensor_type.shape.dim

    channels = None
    input_len = None
    if len(dims) >= 3:
        if dims[1].dim_value > 0:
            channels = int(dims[1].dim_value)
        if dims[2].dim_value > 0:
            input_len = int(dims[2].dim_value)

    return input_name, channels, input_len


def export_representative_data(
    root_path,
    output_path,
    num_samples=256,
    split="train",
    use_gyro=True,
    subject_ids=None,
    onnx_path=None,
    sampling_method="random",
    seed=42,
):
    dataset = MyDataset(
        root_path=root_path,
        split=split,
        subject_ids=subject_ids,
        use_gyro=use_gyro,
    )

    total = len(dataset)
    if total == 0:
        raise ValueError("No samples found for the provided split/subject_ids.")

    sample_count = min(num_samples, total)
    selected_indices = select_indices(
        labels=dataset.labels,
        sample_count=sample_count,
        seed=seed,
        sampling_method=sampling_method,
    )

    x_list, y_list = [], []
    for idx in selected_indices:
        features, label = dataset[idx]
        x_list.append(features.numpy())
        y_list.append(int(label.item()))

    x = np.stack(x_list, axis=0).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)
    subjects = dataset.subjects[selected_indices].astype(np.int64)

    input_name = "input"
    if onnx_path is not None:
        input_name, expected_channels, expected_len = get_onnx_input_spec(onnx_path)
        actual_channels = int(x.shape[1])
        actual_len = int(x.shape[2])

        if expected_channels is not None and actual_channels != expected_channels:
            raise ValueError(
                f"Channel mismatch with ONNX input '{input_name}': expected {expected_channels}, got {actual_channels}."
            )
        if expected_len is not None and actual_len != expected_len:
            raise ValueError(
                f"Input length mismatch with ONNX input '{input_name}': expected {expected_len}, got {actual_len}."
            )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        output_path,
        input=x,
        x=x,
        y=y,
        subjects=subjects,
        indices=selected_indices.astype(np.int64),
        input_name=np.array(input_name),
    )

    print(f"Saved representative data: {output_path}")
    print(f"x shape: {x.shape} | y shape: {y.shape} | subjects shape: {subjects.shape}")
    print(f"Samples exported: {sample_count}/{total}")
    print(f"Sampling method: {sampling_method}")
    label_ids, label_counts = np.unique(y, return_counts=True)
    label_dist = ", ".join(
        [f"{int(label_id)}: {int(count)}" for label_id, count in zip(label_ids, label_counts)]
    )
    print(f"Label distribution (label: count): {label_dist}")
    if onnx_path is not None:
        print(f"Validated against ONNX input: {input_name}")


def main():
    parser = argparse.ArgumentParser(description="Export representative UCI-HAR samples to NPZ")
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="/home/thongp/master/thesis/simulation/master-thesis/uci-har",
        help="Path to UCI-HAR root folder",
    )
    parser.add_argument(
        "--output",
        default="./uci-har/representative_fromtrain.npz",
        type=str,
        help="Output .npz path",
    )
    parser.add_argument("--num-samples", type=int, default=200, help="Number of samples to export")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--sampling-method", type=str, default="random", choices=["random", "stratified"], help="Sampling strategy for representative set")
    parser.add_argument(
        "--onnx-path",
        type=str,
        default="/home/thongp/master/thesis/simulation/master-thesis/models/best_model.onnx",
        help="Optional ONNX model path for shape validation",
    )
    parser.add_argument(
        "--subject-ids",
        type=str,
        default=None,
        help="Comma-separated subject IDs (e.g., '1,3,5'). Use all if omitted.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")

    args = parser.parse_args()

    export_representative_data(
        root_path=args.dataset_path,
        output_path=args.output,
        num_samples=args.num_samples,
        split=args.split,
        sampling_method=args.sampling_method,
        subject_ids=parse_subject_ids(args.subject_ids),
        onnx_path=args.onnx_path,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
