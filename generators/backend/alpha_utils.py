"""Shared alpha-node naming helpers for backend generation."""


def buffer_name_to_alpha_variant(buffer_name: str) -> str:
    """Map a data-model/buffer name to its Rust TensorOp alpha variant."""
    if buffer_name == "d0":
        return "AlphaHBM"
    assert buffer_name.startswith("d"), f"unexpected data model name: {buffer_name}"
    return "Alpha" + buffer_name.upper()


def buffer_name_to_alpha_op(buffer_name: str) -> str:
    """Map a data-model/buffer name to its textual alpha operator."""
    if buffer_name == "d0":
        return "alpha-hbm"
    assert buffer_name.startswith("d"), f"unexpected data model name: {buffer_name}"
    return "alpha-" + buffer_name.replace("_", "-")
