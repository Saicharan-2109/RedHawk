"""Exercise real Streamlit widget reruns, presets, validation, and downloads."""

from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from spatial_lab.image_io import encode_png


APP = Path(__file__).resolve().parents[1] / "app.py"


def launch():
    app = AppTest.from_file(str(APP), default_timeout=45).run()
    assert not app.exception
    return app


def metric_values(app):
    return {metric.label: metric.value for metric in app.metric}


def test_initial_demo_renders_all_outputs_and_downloads():
    app = launch()
    assert app.title[0].value == "See what a matrix can do."
    assert app.selectbox(key="demo_name").value == "Coastal architecture"
    assert metric_values(app)["Processing canvas"] == "700 × 455"
    assert metric_values(app)["Forecast width"] == "+40 px"
    assert len(app.get("download_button")) == 5
    assert not app.error


@pytest.mark.parametrize("name", ["Geometric study", "Light & texture"])
def test_switching_demo_recomputes_without_exception(name):
    app = launch()
    app.selectbox(key="demo_name").set_value(name).run()
    assert not app.exception
    assert not app.error
    assert app.selectbox(key="demo_name").value == name
    assert metric_values(app)["Processing canvas"] == "700 × 455"


def test_upload_mode_without_file_has_clear_empty_state():
    app = launch()
    app.radio(key="source_mode").set_value("Upload image").run()
    assert not app.exception
    assert len(app.get("file_uploader")) == 1
    assert any("Upload a PNG or JPEG to begin" in item.value for item in app.info)
    assert "Processing canvas" not in metric_values(app)
    assert not app.get("download_button")


def test_actual_file_upload_processes_one_row_and_can_be_cleared():
    app = launch()
    app.radio(key="source_mode").set_value("Upload image").run()
    image = np.array([[0, 128, 255]], dtype=np.uint8)
    app.get("file_uploader")[0].set_value(("tiny.png", encode_png(image), "image/png")).run()
    assert not app.exception
    assert not app.error
    assert metric_values(app)["Processing canvas"] == "3 × 1"
    assert len(app.get("download_button")) == 5
    assert not any(slider.key == "profile_row" for slider in app.slider)
    app.get("file_uploader")[0].set_value(None).run()
    assert not app.exception
    assert "Processing canvas" not in metric_values(app)


def test_corrupt_upload_error_and_replacement_recovery():
    app = launch()
    app.radio(key="source_mode").set_value("Upload image").run()
    app.get("file_uploader")[0].set_value(("broken.png", b"bad image", "image/png")).run()
    assert not app.exception
    assert any("Unsupported image data" in error.value for error in app.error)
    assert not app.get("download_button")
    image = np.zeros((5, 8, 3), dtype=np.uint8)
    app.get("file_uploader")[0].set_value(("replacement.png", encode_png(image), "image/png")).run()
    assert not app.exception
    assert not app.error
    assert metric_values(app)["Processing canvas"] == "8 × 5"


def test_resolution_and_extension_controls_affect_results():
    app = launch()
    app.select_slider(key="max_dimension").set_value(350).run()
    app.slider(key="extension_px").set_value(87).run()
    assert not app.exception
    values = metric_values(app)
    assert values["Processing canvas"] == "350 × 227"
    assert values["Forecast width"] == "+87 px"
    app.slider(key="extension_px").set_value(0).run()
    assert not app.exception
    assert metric_values(app)["Forecast width"] == "+0 px"


def test_unstable_parameters_stop_cleanly_and_reset_recovers():
    app = launch()
    app.slider(key="phi_1").set_value(0.95).run()
    assert not app.exception
    assert any("Unstable coefficients" in item.value for item in app.error)
    assert not app.get("download_button")
    app.button(key="reset_parameters").click().run()
    assert not app.exception
    assert not app.error
    assert app.slider(key="phi_1").value == 0.55
    assert metric_values(app)["Forecast width"] == "+40 px"


def test_presets_populate_controls_and_reset_restores_defaults():
    app = launch()
    app.selectbox(key="preset").set_value("Gentle response").run()
    assert not app.exception
    assert app.slider(key="phi_1").value == 0.4
    assert app.slider(key="edge_gain").value == 2.0
    app.slider(key="extension_px").set_value(90).run()
    app.select_slider(key="max_dimension").set_value(350).run()
    app.button(key="reset_parameters").click().run()
    assert not app.exception
    assert app.selectbox(key="preset").value == "Blueprint defaults"
    assert app.slider(key="phi_1").value == 0.55
    assert app.slider(key="edge_gain").value == 4.0
    assert app.slider(key="extension_px").value == 40
    assert app.select_slider(key="max_dimension").value == 700
