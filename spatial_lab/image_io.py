"""Validated image inputs, deterministic demonstration images, and portable exports."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from functools import lru_cache
from io import BytesIO
import json
from typing import TYPE_CHECKING
import warnings
import zipfile

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

if TYPE_CHECKING:
    from spatial_lab.engine import EngineParameters, TransformResult


MAX_UPLOAD_BYTES = 15 * 1024 * 1024
MAX_SOURCE_PIXELS = 40_000_000
DEMO_NAMES = ("Coastal architecture", "Geometric study", "Light & texture")


class ImageValidationError(ValueError):
    """An image could not be safely decoded or prepared for this engine."""


@dataclass(frozen=True)
class PreparedImage:
    """RGB preview and grayscale input at the same, bounded processing resolution."""

    rgb: np.ndarray
    gray: np.ndarray
    original_width: int
    original_height: int
    source_name: str
    resized: bool

    @property
    def width(self) -> int:
        return int(self.gray.shape[1])

    @property
    def height(self) -> int:
        return int(self.gray.shape[0])


def _validate_dimension(max_dimension: int) -> None:
    if isinstance(max_dimension, bool) or not isinstance(max_dimension, int):
        raise ImageValidationError("Processing resolution must be a whole number.")
    if not 128 <= max_dimension <= 1600:
        raise ImageValidationError("Choose a processing resolution between 128 and 1600 pixels.")


def decode_image(data: bytes, source_name: str, max_dimension: int = 700) -> PreparedImage:
    """Read a PNG/JPEG, apply EXIF orientation, composite alpha, then downscale.

    Pillow reads dimensions before decoding pixel data, so oversized sources are
    rejected early. Sixteen-bit grayscale PNG levels use their fixed 0..65535
    range, avoiding Pillow's clipping during direct RGB conversion. Upload bytes
    remain in memory and are never written to disk.
    """
    _validate_dimension(max_dimension)
    if not isinstance(data, bytes) or not data:
        raise ImageValidationError("The image is empty. Choose a PNG or JPEG file.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise ImageValidationError("This file exceeds 15 MiB. Choose a smaller PNG or JPEG.")
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        expected_format = "PNG"
    elif data.startswith(b"\xff\xd8\xff"):
        expected_format = "JPEG"
    else:
        raise ImageValidationError("Unsupported image data. Upload a real PNG or JPEG file.")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(data)) as source:
                if source.format != expected_format:
                    raise ImageValidationError("The image contents do not match a PNG or JPEG.")
                if source.width * source.height > MAX_SOURCE_PIXELS:
                    raise ImageValidationError("The image exceeds 40 megapixels. Resize it before uploading.")
                if getattr(source, "is_animated", False):
                    raise ImageValidationError("Choose a still PNG or JPEG instead of an animated image.")
                source.load()
                oriented = ImageOps.exif_transpose(source)
                original_width, original_height = oriented.size
                if original_width < 3:
                    raise ImageValidationError("The image must be at least 3 pixels wide for spatial lags.")
                if expected_format == "PNG" and (
                    oriented.mode == "I" or oriented.mode.startswith("I;16")
                ):
                    levels = np.asarray(oriented)
                    gray8 = np.rint(np.clip(levels, 0, 65535).astype(np.float64) / 257).astype(np.uint8)
                    transparent_level = oriented.info.get("transparency")
                    if isinstance(transparent_level, int):
                        gray8[levels == transparent_level] = 255
                    rgb = np.repeat(gray8[:, :, None], 3, axis=2)
                elif oriented.mode in ("RGBA", "LA") or "transparency" in oriented.info:
                    transparent = oriented.convert("RGBA")
                    background = Image.new("RGBA", transparent.size, (255, 255, 255, 255))
                    rgb = np.asarray(Image.alpha_composite(background, transparent).convert("RGB")).copy()
                else:
                    rgb = np.asarray(oriented.convert("RGB")).copy()
    except ImageValidationError:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ImageValidationError("The image dimensions are too large. Resize the source before uploading.") from exc
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        raise ImageValidationError("The image is damaged or incomplete. Try another PNG or JPEG.") from exc

    resized = max(original_width, original_height) > max_dimension
    if resized:
        scale = max_dimension / max(original_width, original_height)
        target_width = max(1, round(original_width * scale))
        target_height = max(1, round(original_height * scale))
        if target_width < 3:
            raise ImageValidationError(
                "This image becomes narrower than 3 pixels at the selected resolution. "
                "Increase the resolution or choose a less narrow image."
            )
        rgb = cv2.resize(rgb, (target_width, target_height), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    safe_name = str(source_name).replace("\\", "/").rsplit("/", 1)[-1] or "image.png"
    return PreparedImage(
        rgb=np.ascontiguousarray(rgb),
        gray=np.ascontiguousarray(gray),
        original_width=original_width,
        original_height=original_height,
        source_name=safe_name,
        resized=resized,
    )


def encode_png(image: np.ndarray) -> bytes:
    """Encode a uint8 grayscale or RGB image as a lossless PNG."""
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or not image.size:
        raise ImageValidationError("PNG export requires a nonempty uint8 image.")
    if image.ndim == 2:
        output = image
    elif image.ndim == 3 and image.shape[2] == 3:
        output = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        raise ImageValidationError("PNG export supports grayscale and RGB images.")
    try:
        success, encoded = cv2.imencode(".png", np.ascontiguousarray(output))
    except cv2.error as exc:
        raise ImageValidationError("Could not encode this image as PNG.") from exc
    if not success:
        raise ImageValidationError("Could not encode this image as PNG.")
    return encoded.tobytes()


def _gradient(height: int, width: int, top: tuple, bottom: tuple) -> np.ndarray:
    ratio = np.linspace(0, 1, height, dtype=np.float32)[:, None, None]
    color = (1 - ratio) * np.asarray(top) + ratio * np.asarray(bottom)
    return np.broadcast_to(color, (height, width, 3)).copy().astype(np.uint8)


@lru_cache(maxsize=3)
def _demo_bytes(name: str) -> bytes:
    """Draw original, reproducible scenes that expose edges, shading, and texture."""
    height, width = 650, 1000
    rng = np.random.default_rng(311)
    if name == "Coastal architecture":
        canvas = _gradient(height, width, (112, 174, 194), (226, 225, 204))
        cv2.circle(canvas, (822, 125), 53, (248, 230, 186), -1, cv2.LINE_AA)
        cv2.rectangle(canvas, (0, 406), (999, 650), (61, 126, 144), -1)
        for y in range(418, 650, 18):
            cv2.line(canvas, (0, y), (999, y - 4), (102, 158, 164), 1, cv2.LINE_AA)
        cv2.fillPoly(canvas, [np.array([(0, 470), (520, 444), (1000, 606), (1000, 650), (0, 650)])], (214, 198, 172))
        cv2.rectangle(canvas, (142, 162), (586, 499), (227, 220, 195), -1)
        cv2.fillPoly(canvas, [np.array([(586, 162), (713, 224), (713, 528), (586, 499)])], (179, 172, 147))
        cv2.rectangle(canvas, (124, 145), (602, 165), (245, 235, 207), -1)
        cv2.line(canvas, (602, 155), (727, 218), (211, 201, 171), 18, cv2.LINE_AA)
        for x in (181, 316, 451):
            cv2.rectangle(canvas, (x - 7, 227), (x + 80, 354), (246, 238, 212), -1)
            cv2.rectangle(canvas, (x, 234), (x + 72, 347), (39, 71, 76), -1)
            cv2.line(canvas, (x + 36, 235), (x + 36, 347), (172, 180, 160), 3)
            cv2.line(canvas, (x, 291), (x + 72, 291), (172, 180, 160), 3)
        cv2.ellipse(canvas, (363, 455), (49, 85), 0, 180, 360, (129, 98, 73), -1, cv2.LINE_AA)
        cv2.rectangle(canvas, (314, 455), (412, 498), (129, 98, 73), -1)
        for x in range(320, 409, 10):
            cv2.line(canvas, (x, 421), (x, 497), (151, 120, 85), 2, cv2.LINE_AA)
        cv2.line(canvas, (770, 510), (785, 306), (94, 100, 69), 10, cv2.LINE_AA)
        for x, y in [(706, 330), (758, 297), (820, 323), (851, 359), (728, 372), (803, 385)]:
            cv2.ellipse(canvas, (x, y), (58, 23), -20 if x < 785 else 25, 0, 360, (67, 109, 81), -1, cv2.LINE_AA)
        cv2.rectangle(canvas, (100, 498), (617, 511), (188, 176, 148), -1)
    elif name == "Geometric study":
        canvas = _gradient(height, width, (241, 235, 219), (207, 200, 180))
        cv2.fillPoly(canvas, [np.array([(0, 525), (1000, 352), (1000, 650), (0, 650)])], (192, 183, 161))
        cv2.fillPoly(canvas, [np.array([(120, 474), (448, 581), (906, 504), (593, 405)])], (154, 151, 132))
        cv2.rectangle(canvas, (196, 239), (432, 472), (195, 113, 77), -1)
        cv2.fillPoly(canvas, [np.array([(432, 239), (529, 188), (529, 422), (432, 472)])], (142, 77, 55))
        cv2.fillPoly(canvas, [np.array([(196, 239), (294, 188), (529, 188), (432, 239)])], (226, 153, 112))
        cv2.circle(canvas, (650, 364), 126, (72, 108, 105), -1, cv2.LINE_AA)
        yy, xx = np.mgrid[:height, :width]
        sphere = (xx - 650) ** 2 + (yy - 364) ** 2 <= 126 ** 2
        shade = np.clip(1.05 - (xx - 560) / 320 - (yy - 260) / 720, 0.35, 1.25)
        canvas[sphere] = np.clip(np.array([96, 145, 133]) * shade[sphere, None], 0, 255).astype(np.uint8)
        cv2.rectangle(canvas, (716, 179), (790, 281), (221, 190, 120), -1)
        cv2.ellipse(canvas, (753, 179), (37, 17), 0, 0, 360, (243, 218, 157), -1, cv2.LINE_AA)
        cv2.ellipse(canvas, (753, 281), (37, 17), 0, 0, 180, (221, 190, 120), -1, cv2.LINE_AA)
        for x in range(53, 950, 34):
            cv2.line(canvas, (x, 56), (x, 82), (158, 158, 138), 1, cv2.LINE_AA)
    elif name == "Light & texture":
        yy, xx = np.mgrid[:height, :width]
        field = 72 + 137 * (xx / width) + 25 * np.sin(xx / 49) * np.cos(yy / 77)
        grain = rng.normal(0, 6, (height, width))
        base = np.clip(field + grain, 0, 255)
        canvas = np.stack([base * 1.07, base, base * .85], axis=-1).clip(0, 255).astype(np.uint8)
        for x in range(-500, 1500, 127):
            cv2.fillPoly(canvas, [np.array([(x, 0), (x + 46, 0), (x + 351, 650), (x + 305, 650)])], (81, 83, 70))
        cv2.rectangle(canvas, (317, 149), (624, 493), (219, 211, 184), -1)
        for y in range(157, 490, 9):
            cv2.line(canvas, (321, y), (620, y), (197, 188, 161), 1)
        cv2.circle(canvas, (472, 320), 106, (152, 94, 66), -1, cv2.LINE_AA)
        cv2.circle(canvas, (472, 320), 73, (216, 195, 154), -1, cv2.LINE_AA)
        cv2.circle(canvas, (472, 320), 42, (79, 89, 73), -1, cv2.LINE_AA)
        for y in range(530, 650, 8):
            cv2.line(canvas, (0, y), (999, y), (118, 119, 99), 1, cv2.LINE_AA)
    else:
        raise ImageValidationError(f"Unknown demonstration image. Choose one of: {', '.join(DEMO_NAMES)}.")
    return encode_png(canvas)


def demo_image(name: str, max_dimension: int = 700) -> PreparedImage:
    """Return an original procedural sample through the same upload pipeline."""
    return decode_image(_demo_bytes(name), name, max_dimension)


def build_download_bundle(
    prepared: PreparedImage, result: TransformResult, params: EngineParameters
) -> bytes:
    """Export PNGs, numeric arrays, and a manifest sufficient to repeat the run.

    source.png is the orientation-corrected RGB image at processing resolution;
    the original uploaded bytes are deliberately not retained in the archive.
    """
    manifest = {
        "application": "Matrix Vision Lab",
        "format_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_name": prepared.source_name,
        "original_dimensions": {"width": prepared.original_width, "height": prepared.original_height},
        "processed_dimensions": {"width": prepared.width, "height": prepared.height},
        "extension_dimensions": {"width": int(result.extension.shape[1]), "height": int(result.extension.shape[0])},
        "source_png_note": "Oriented RGB image at processing resolution, not original upload bytes.",
        "parameters": asdict(params),
        "spectral_radius": float(result.operators.spectral_radius),
        "condition_number": float(result.operators.condition_number),
        "elapsed_ms": float(result.elapsed_ms),
        "boundary_policy": "Replicate the first pixel for unavailable left lags.",
        "model_note": "Deterministic spatial state-space demonstration; no fitted ARIMA or learned inpainting model.",
    }
    arrays = BytesIO()
    np.savez_compressed(
        arrays,
        source_rgb=prepared.rgb,
        grayscale=prepared.gray,
        edges=result.edges,
        baseline=result.baseline,
        extension=result.extension,
        transition=result.operators.transition,
        difference=result.operators.difference,
        integration=result.operators.integration,
    )
    output = BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, pixels in (
            ("source.png", prepared.rgb),
            ("grayscale.png", prepared.gray),
            ("edges.png", result.edges),
            ("baseline.png", result.baseline),
            ("extension.png", result.extension),
        ):
            archive.writestr(name, encode_png(pixels))
        archive.writestr("matrices.npz", arrays.getvalue())
        archive.writestr("manifest.json", json.dumps(manifest, indent=2, allow_nan=False))
    return output.getvalue()
