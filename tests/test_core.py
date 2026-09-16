import torch
import pytest
from PIL import Image

from evaluate import select_indices
from unet_learning import UNet, lab01_to_rgb01_differentiable
from YT.inference_colorize import colorize_image, load_generator


def test_selection_is_deterministic() -> None:
    first = select_indices(1_000, 100, 42)
    second = select_indices(1_000, 100, 42)
    assert first == second
    assert len(first) == 100
    assert len(set(first)) == 100


def test_checkpoint_load_is_strict(tmp_path) -> None:
    checkpoint = tmp_path / "model.pt"
    model = UNet(in_channels=1, out_channels=2, features=4)
    torch.save({"G": model.state_dict(), "epoch": 1}, checkpoint)

    loaded = load_generator(checkpoint, features=4, device=torch.device("cpu"))
    assert isinstance(loaded, UNet)

    with pytest.raises(ValueError, match="incompatible"):
        load_generator(checkpoint, features=8, device=torch.device("cpu"))


def test_colorization_preserves_original_dimensions(tmp_path) -> None:
    checkpoint = tmp_path / "model.pt"
    model = UNet(in_channels=1, out_channels=2, features=4)
    torch.save({"G": model.state_dict()}, checkpoint)
    loaded = load_generator(checkpoint, features=4, device=torch.device("cpu"))

    source = Image.new("L", (19, 17), color=128)
    result = colorize_image(
        loaded,
        source,
        torch.device("cpu"),
        max_size=1024,
        boost_saturation=1.0,
    )
    assert result.size == source.size
    assert result.mode == "RGB"


def test_lab_conversion_keeps_nonzero_gradient() -> None:
    luminance = torch.full((1, 1, 8, 8), 0.5)
    chroma = torch.full((1, 2, 8, 8), 0.5, requires_grad=True)
    rgb = lab01_to_rgb01_differentiable(luminance, chroma)
    rgb.mean().backward()

    assert chroma.grad is not None
    assert torch.count_nonzero(chroma.grad).item() > 0
