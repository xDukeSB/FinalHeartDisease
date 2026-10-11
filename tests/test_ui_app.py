"""UI test simulating Streamlit application interactions."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).parent.parent / "app.py")


def test_app_initial_render():
    """Verify the Streamlit app loads without exceptions and displays checkup controls."""
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    assert not at.exception, f"App raised an exception on load: {at.exception}"

    # Verify action buttons exist
    button_labels = [b.label for b in at.button]
    assert any("Analyze" in b for b in button_labels)
    assert any("Reset Form" in b for b in button_labels)

    # Verify checkup input fields
    number_labels = [n.label for n in at.number_input]
    assert any("Age" in n for n in number_labels)
    assert any("Systolic" in n for n in number_labels)
    assert any("Diastolic" in n for n in number_labels)
    assert any("Cholesterol" in n for n in number_labels)
    assert any("HDL" in n for n in number_labels)
    assert any("Height" in n for n in number_labels)
    assert any("Weight" in n for n in number_labels)

    selectbox_labels = [s.label for s in at.selectbox]
    assert any("Sex" in s for s in selectbox_labels)
    assert any("Smoking" in s for s in selectbox_labels)
    assert any("Diabetes" in s for s in selectbox_labels)
    assert any("Family History" in s for s in selectbox_labels)
    assert any("Activity" in s for s in selectbox_labels)


def test_app_analyze_flow():
    """Simulate user submitting the form to get a prediction result."""
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()

    # Click 'Analyze Health Checkup Report'
    analyze_btn = next(b for b in at.button if "Analyze" in b.label)
    analyze_btn.click().run()

    assert not at.exception
    assert at.session_state.get("prediction_result") is not None
    res = at.session_state["prediction_result"]
    assert "prediction_class" in res
    assert "risk_level" in res
    assert "probability_disease" in res
    assert "narrative" in res


def test_app_reset_flow():
    """Simulate clicking Reset Form and verifying state clears."""
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()

    # Analyze first
    analyze_btn = next(b for b in at.button if "Analyze" in b.label)
    analyze_btn.click().run()
    assert at.session_state.get("prediction_result") is not None

    # Now click Reset
    reset_btn = next(b for b in at.button if "Reset Form" in b.label)
    reset_btn.click().run()

    assert not at.exception
    assert at.session_state.get("prediction_result") is None


def test_multiple_submissions_and_reruns():
    """Verify that multiple submissions and widget state changes do not raise ValueError."""
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()

    # First submission
    analyze_btn = next(b for b in at.button if "Analyze" in b.label)
    analyze_btn.click().run()
    assert not at.exception

    # Second submission
    analyze_btn = next(b for b in at.button if "Analyze" in b.label)
    analyze_btn.click().run()
    assert not at.exception
