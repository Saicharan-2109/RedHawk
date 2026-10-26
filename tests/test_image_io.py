"""Image decoding, limits, orientation and reproducible download contracts."""

from dataclasses import asdict
from io import BytesIO
import json
import struct
import zipfile
import zlib

import cv2
import numpy as np
from PIL import Image
import pytest

from spatial_lab.engine import EngineParameters, process_image
from spatial_lab.image_io import (
    DEMO_NAMES,
    MAX_UPLOAD_BYTES,
    ImageValidationError,
    build_download_bundle,
    decode_image,
    demo_image,
    encode_png,
)


def pillow_bytes(image, format="PNG", **options):
    stream = BytesIO()
    image.save(stream, format=format, **options)
    return stream.getvalue()


def test_rgb_png_roundtrip_uses_correct_channel_order():
    rgb = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255]]], dtype=np.uint8)
    prepared = decode_image(encode_png(rgb), "colors.png")
    np.testing.assert_array_equal(prepared.rgb, rgb)
    np.testing.assert_array_equal(prepared.gray, cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
    assert (prepared.width, prepared.height) == (3, 1)
    assert not prepared.resized


def test_grayscale_png_and_jpeg_load():
    gray = np.tile(np.arange(64, dtype=np.uint8) * 4, (32, 1))
    prepared = decode_image(encode_png(gray), "gray.png")
    np.testing.assert_array_equal(prepared.gray, gray)
    jpeg = pillow_bytes(Image.fromarray(gray), "JPEG", quality=98)
    result = decode_image(jpeg, "gray.jpg")
    assert result.gray.shape == gray.shape
    np.testing.assert_allclose(result.gray.astype(int), gray.astype(int), atol=2)


def test_16bit_grayscale_png_scales_full_range_without_white_saturation():
    levels = np.array([[0, 16384, 32768, 65535]], dtype=np.uint16)
    prepared = decode_image(pillow_bytes(Image.fromarray(levels)), "16bit.png")
    np.testing.assert_array_equal(prepared.gray, [[0, 64, 128, 255]])


def test_16bit_grayscale_transparency_is_white():
    levels = np.array([[0, 32768, 65535]], dtype=np.uint16)
    prepared = decode_image(pillow_bytes(Image.fromarray(levels), transparency=32768), "16bit-alpha.png")
    np.testing.assert_array_equal(prepared.gray, [[0, 255, 255]])


def test_transparency_is_composited_onto_white():
    image = Image.new("RGBA", (4, 2), (10, 20, 30, 0))
    image.putpixel((1, 0), (255, 0, 0, 255))
    prepared = decode_image(pillow_bytes(image), "transparent.png")
    np.testing.assert_array_equal(prepared.rgb[0, 0], [255, 255, 255])
    np.testing.assert_array_equal(prepared.rgb[0, 1], [255, 0, 0])


def test_exif_rotation_precedes_metadata_and_processing():
    image = Image.new("RGB", (7, 3), (90, 110, 140))
    exif = Image.Exif()
    exif[274] = 6  # Rotate 90 degrees clockwise for display.
    prepared = decode_image(pillow_bytes(image, "JPEG", exif=exif), "rotated.jpg")
    assert (prepared.original_width, prepared.original_height) == (3, 7)
    assert prepared.rgb.shape == (7, 3, 3)


def test_resize_preserves_aspect_ratio_and_small_sources_are_not_upscaled():
    large = decode_image(pillow_bytes(Image.new("RGB", (1000, 500))), "large.png", 128)
    assert (large.width, large.height) == (128, 64)
    assert (large.original_width, large.original_height) == (1000, 500)
    assert large.resized
    small = decode_image(pillow_bytes(Image.new("RGB", (10, 5))), "small.png", 128)
    assert (small.width, small.height) == (10, 5)
    assert not small.resized


@pytest.mark.parametrize("data", [b"", b"not an image", b"\x89PNG\r\n\x1a\ntruncated", b"\xff\xd8\xffbroken"])
def test_invalid_or_truncated_images_raise_friendly_error(data):
    with pytest.raises(ImageValidationError):
        decode_image(data, "invalid.png")


def test_upload_byte_limit_rejected_before_decode():
    with pytest.raises(ImageValidationError, match="15 MiB"):
        decode_image(b"x" * (MAX_UPLOAD_BYTES + 1), "huge.png")


def test_source_pixel_limit_rejected_from_header_before_full_decode():
    original = pillow_bytes(Image.new("RGB", (3, 3)))
    # Construct a valid IHDR advertising 48MP while retaining a tiny payload.
    header = struct.pack(">II", 8000, 6000) + original[24:29]
    crc = struct.pack(">I", zlib.crc32(b"IHDR" + header) & 0xFFFFFFFF)
    oversized = original[:16] + header + crc + original[33:]
    with pytest.raises(ImageValidationError, match="40 megapixels"):
        decode_image(oversized, "oversized.png")


@pytest.mark.parametrize("resolution", [127, 1601, 700.5, True])
def test_invalid_processing_resolution(resolution):
    with pytest.raises(ImageValidationError):
        decode_image(encode_png(np.zeros((2, 3), dtype=np.uint8)), "valid.png", resolution)


def test_narrow_images_rejected_before_and_after_downscale():
    with pytest.raises(ImageValidationError, match="3 pixels"):
        decode_image(pillow_bytes(Image.new("RGB", (2, 20))), "narrow.png")
    with pytest.raises(ImageValidationError, match="3 pixels"):
        decode_image(pillow_bytes(Image.new("RGB", (3, 1000))), "tall.png", 128)


@pytest.mark.parametrize("name", DEMO_NAMES)
def test_demo_images_are_deterministic_and_processed_like_uploads(name):
    first = demo_image(name, 350)
    second = demo_image(name, 350)
    assert (first.width, first.height) == (350, 227)
    assert first.resized
    assert first.rgb.dtype == first.gray.dtype == np.uint8
    assert np.ptp(first.gray) > 50
    np.testing.assert_array_equal(first.rgb, second.rgb)


def test_unknown_sample_and_invalid_export_fail_clearly():
    with pytest.raises(ImageValidationError):
        demo_image("not a demo")
    with pytest.raises(ImageValidationError):
        encode_png(np.zeros((3, 3), dtype=float))
    with pytest.raises(ImageValidationError):
        encode_png(np.zeros((3, 3, 4), dtype=np.uint8))


def test_animated_png_is_rejected():
    first = Image.new("RGB", (5, 5), (255, 0, 0))
    second = Image.new("RGB", (5, 5), (0, 0, 255))
    data = pillow_bytes(first, save_all=True, append_images=[second], duration=100, loop=0)
    with pytest.raises(ImageValidationError, match="still PNG"):
        decode_image(data, "animated.png")


def test_download_bundle_images_arrays_parameters_and_safe_filenames():
    prepared = decode_image(encode_png(np.arange(48, dtype=np.uint8).reshape(6, 8)), "../../image.png")
    params = EngineParameters(extension_px=9)
    result = process_image(prepared.gray, params)
    data = build_download_bundle(prepared, result, params)
    with zipfile.ZipFile(BytesIO(data)) as archive:
        assert set(archive.namelist()) == {
            "source.png", "grayscale.png", "edges.png", "baseline.png", "extension.png",
            "matrices.npz", "manifest.json",
        }
        assert all("/" not in name and "\\" not in name for name in archive.namelist())
        manifest = json.loads(archive.read("manifest.json"))
        assert manifest["source_name"] == "image.png"
        assert manifest["parameters"] == asdict(params)
        assert manifest["processed_dimensions"] == {"width": 8, "height": 6}
        assert manifest["extension_dimensions"] == {"width": 17, "height": 6}
        arrays = np.load(BytesIO(archive.read("matrices.npz")), allow_pickle=False)
        np.testing.assert_array_equal(arrays["grayscale"], prepared.gray)
        np.testing.assert_array_equal(arrays["extension"], result.extension)
        np.testing.assert_array_equal(arrays["transition"], result.operators.transition)
        for name in ("edges", "baseline", "extension"):
            png = cv2.imdecode(np.frombuffer(archive.read(f"{name}.png"), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            np.testing.assert_array_equal(png, getattr(result, name))
