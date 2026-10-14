"""Local, privacy-aware Streamlit interface for the locked WheatPheno models.

Start from the repository root with ``streamlit run app.py``. Model loading,
feature validation and inference belong to wheatpheno, not to this interface.
"""
from __future__ import annotations

import io
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
TARGET_LABELS = {
    "AGB": "Aboveground biomass (AGB)",
    "Yield": "Grain yield",
    "PH": "Plant height (PH)",
}
SCENES = {}  # No field scenes are distributed in this release.
SESSION_VERSION = "0.4.0-code-first"
MAX_UPLOAD_BYTES = 32 * 1024 * 1024

st.set_page_config(
    page_title="WheatPheno | Code and Models",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(
    """<style>
    .stApp {background: #f7f9fb;}
    .block-container {padding-top: 2.1rem; padding-bottom: 3rem; max-width: 1420px;}
    [data-testid="stSidebar"] {background: #eaf0ee;}
    h1, h2, h3 {color: #173f39; letter-spacing: -.025em;}
    h1 {font-size: 2.55rem !important;}
    [data-testid="stMetric"] {background: white; border: 1px solid #dfe7e4;
        padding: 1.1rem; border-radius: 12px;}
    [data-testid="stMetricValue"] {color: #1c6254;}
    div.stButton > button[kind="primary"] {background: #236d5d; border-color: #236d5d;}
    .eyebrow {color: #47766d; letter-spacing: .14em; font-size: .78rem;
        font-weight: 700; margin-bottom: .45rem;}
    .subtitle {font-size: 1.04rem; color: #596a68; max-width: 860px; line-height: 1.8;}
    .note {color: #64736e; font-size: .88rem; line-height: 1.7;}
    </style>""",
    unsafe_allow_html=True,
)


def load_api() -> dict[str, Any]:
    """Delay project imports so a missing installation has a useful UI message."""
    try:
        from wheatpheno.models import load_model, model_manifest
        from wheatpheno.pipeline import predict_csv
        from wheatpheno.schema import feature_names
    except ImportError as exc:
        st.error("WheatPheno dependencies are unavailable. Follow the installation steps in the README.")
        st.code('python -m pip install -e ".[ui]"\nstreamlit run app.py', language="bash")
        with st.expander("Import details"):
            st.text(str(exc))
        st.stop()
    return {
        "manifest": model_manifest,
        "load_model": load_model,
        "features": feature_names,
        "predict_csv": predict_csv,
    }


def readable_manifest(api: dict[str, Any]) -> dict[str, Any]:
    source = api["manifest"]
    try:
        value = source() if callable(source) else source
        return value if isinstance(value, dict) else {"manifest": value}
    except Exception as exc:
        return {"status": "unavailable", "detail": str(exc)}


def required_features(api: dict[str, Any], target: str) -> list[str]:
    if target == "PH":
        return ["CHM_PCT95"]
    value = api["features"]
    if callable(value):
        try:
            value = value(target)
        except TypeError:
            value = value()
    if isinstance(value, dict):
        value = value.get(target, value.get(target.lower(), value.get("features", [])))
    return [str(item) for item in value] if isinstance(value, (list, tuple)) else []


def safe_display(value: Any) -> Any:
    """Do not expose original workstation locations in reports shown in the UI."""
    if isinstance(value, dict):
        return {str(key): safe_display(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_display(item) for item in value]
    if isinstance(value, Path):
        value = str(value)
    if isinstance(value, str):
        # Only redact whole absolute paths, not explanatory sentences or URLs.
        candidate = Path(value)
        if candidate.is_absolute() and not value.startswith(("http://", "https://")):
            try:
                return str(candidate.resolve().relative_to(ROOT))
            except (ValueError, OSError):
                return "[local]/" + candidate.name
    return value


def json_bytes(value: Any) -> bytes:
    return json.dumps(safe_display(value), ensure_ascii=False, indent=2, default=str).encode("utf-8")


def new_run(prefix: str) -> Path:
    # Every session gets isolated local output; public examples are never rewritten.
    return Path(tempfile.mkdtemp(prefix=f"wheatpheno_{prefix}_"))


def display_reports(root: Path, key: str) -> None:
    files = sorted(root.rglob("*report*.json"))
    for index, path in enumerate(files):
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        with st.expander(f"Quality report · {path.relative_to(root).as_posix()}"):
            st.json(safe_display(report))
            st.download_button(
                "Download report (JSON)",
                json_bytes(report),
                file_name=f"{key}_{index}_report.json",
                mime="application/json",
                key=f"report_dl_{key}_{index}",
            )


def page_overview(api: dict[str, Any], manifest: dict[str, Any]) -> None:
    st.markdown('<div class="eyebrow">WHEAT PHENOTYPING · OPEN CODE AND LIMITED DATA</div>', unsafe_allow_html=True)
    st.title("WheatPheno code and models")
    st.markdown(
        '<div class="subtitle">WheatPheno provides reusable Python code, command line workflows, documented feature definitions '
        'and fixed regression models. This optional interface supports table prediction and model inspection.</div>',
        unsafe_allow_html=True,
    )
    st.write("")
    col1, col2, col3 = st.columns(3)
    col1.metric("Target traits", "PH · AGB · Yield")
    col2.metric("Regression model", "Extra Trees")
    col3.metric("Public table examples", "21 records")
    st.write("")
    left, right = st.columns([1.25, 1])
    with left:
        st.subheader("Get started")
        st.markdown(
            "1. **Use the code**: follow the README for feature preparation, model training and command line inference.\n"
            "2. **Reproduce fixed predictions**: run the supplied table examples or upload a compatible CSV.\n"
            "3. **Prepare your own maps**: use the Spatial Mapping page for the verified raster workflow."
        )
        st.info("Mapping does not apply an OSAVI threshold mask. OSAVI remains a model input; boundary and input-quality checks still apply.")
    with right:
        st.subheader("Release scope")
        st.markdown(
            "The release includes code, fixed models, 12 AGB records, nine yield records and synthetic examples. "
            "Field imagery and field scene archives are excluded.\n\n"
            "The 21 observed records demonstrate fixed inference and overlap model training. "
            "Synthetic examples demonstrate algorithms and do not measure agricultural prediction accuracy."
        )
        st.caption("Files are processed on the computer running this application. A hosted deployment uploads files to its server.")
    with st.expander("Model manifest"):
        st.json(safe_display(manifest))
    if st.button("Verify model integrity"):
        try:
            with st.spinner("Checking artifact hashes, runtime compatibility and feature specifications…"):
                for target in ("AGB", "Yield"):
                    api["load_model"](target)
            st.session_state["integrity_check"] = {"targets": ["AGB", "Yield"], "status": "passed", "at_utc": datetime.now(timezone.utc).isoformat()}
            st.success("Both fixed models passed the artifact integrity, runtime and input specification checks.")
        except Exception as exc:
            st.session_state["integrity_check"] = {"status": "failed", "detail": str(exc)}
            st.error(f"Verification failed: {exc}")
    st.caption("The web interface uses the bundled models. New-model deployment and DSM/DTM height workflows are available through the command line; see docs/DEPLOYMENT.md and docs/HEIGHT.md.")


def page_csv(api: dict[str, Any]) -> None:
    st.title("Table Prediction")
    st.caption("Predict from a CSV using the documented feature names and units.")
    target = st.selectbox("Target trait", list(TARGET_LABELS), format_func=TARGET_LABELS.get, key="csv_target")
    names = required_features(api, target)
    if target == "PH":
        st.info("Table-based height estimation converts CHM_PCT95 from metres to centimetres: PH = CHM_PCT95 × 100.")
    else:
        st.caption("AGB and grain yield are reported in g m⁻². Input features must use the same definitions and scales as the training data.")

    source = st.radio("Data source", ["Public example", "Upload CSV"], horizontal=True)
    content: bytes | None = None
    name = "input.csv"
    if source == "Public example":
        sample = ROOT / "examples/tables" / ("yield_demo.csv" if target == "Yield" else "agb_demo.csv")
        if sample.exists():
            content, name = sample.read_bytes(), sample.name
            st.caption("This limited example demonstrates the workflow. It is neither the complete study dataset nor an independent validation set.")
            st.download_button("Download example CSV", content, file_name=name, mime="text/csv", key=f"sample_{target}")
        else:
            st.warning("The example file is unavailable. You can upload a compatible CSV.")
    else:
        upload = st.file_uploader("Choose a CSV file (maximum 32 MB)", type=["csv"], key="csv_upload")
        if upload is not None:
            if upload.size > MAX_UPLOAD_BYTES:
                st.error("The file exceeds 32 MB. Split the table or use the command-line interface for batch prediction.")
                return
            content, name = upload.getvalue(), Path(upload.name).name
    if content is None:
        with st.expander("Required input columns", expanded=True):
            st.write(names or "See Model Specification for the complete feature list.")
        return

    try:
        frame = pd.read_csv(io.BytesIO(content))
    except Exception as exc:
        st.error(f"Could not read CSV: {exc}")
        return
    if frame.empty:
        st.error("The CSV contains no data rows.")
        return
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", len(frame))
    c2.metric("Input columns", len(frame.columns))
    missing = [column for column in names if column not in frame.columns]
    c3.metric("Missing required columns", len(missing) if names else "Not checked")
    with st.expander("Input preview", expanded=True):
        st.dataframe(frame.head(30), width="stretch", hide_index=True)
    if missing:
        st.warning("Required columns are missing: " + ", ".join(missing))
    with st.expander("Features and missing values"):
        st.write(names)
        st.caption("Missing values use the stored preprocessing learned during training. Imputation cannot correct wrong units or rows with no valid feature information.")

    if st.button("Run prediction", type="primary"):
        work = new_run("csv")
        input_path = work / "input.csv"
        input_path.write_bytes(content)
        output_dir = work / "outputs"
        output_dir.mkdir()
        try:
            with st.spinner("Checking inputs and running the fixed model…"):
                result = api["predict_csv"](input_path, output_dir, target)
            st.session_state["csv_result"] = {
                "dir": str(output_dir), "target": target, "source": name,
                "result": safe_display(result), "at": datetime.now(timezone.utc).isoformat(),
            }
            st.success("Prediction complete.")
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
    state = st.session_state.get("csv_result")
    if state:
        output_dir = Path(state["dir"])
        st.subheader("Latest results")
        st.caption(f"Target: {TARGET_LABELS[state['target']]} · Source: {state['source']}")
        files = sorted(output_dir.rglob("*.csv"))
        for index, path in enumerate(files):
            try:
                result_frame = pd.read_csv(path)
                st.dataframe(result_frame.head(50), width="stretch", hide_index=True)
            except (OSError, ValueError):
                continue
            st.download_button("Download predictions (CSV)", path.read_bytes(), path.name, "text/csv", key=f"prediction_csv_{index}")
        if not files:
            st.json(state["result"])
        display_reports(output_dir, "csv")


def page_mapping(api: dict[str, Any]) -> None:
    st.title("Spatial Mapping")
    st.caption("Apply the command line workflow to your own verified feature rasters.")
    st.markdown(
        "The public release contains no field scene archives or field maps. "
        "Spatial inference is available through the Python API and command line tools. "
        "Prepare source coverage, a field boundary and feature layers with the definitions "
        "and units required by the fixed models."
    )
    st.subheader("Prepare and map your rasters")
    st.code(
        "python -m wheatpheno pack --config configs/rasters.example.json --output outputs/prepared\n"
        "python -m wheatpheno map --config outputs/prepared/scene.json --output outputs/mapped",
        language="bash",
    )
    st.caption("Replace the raster configuration paths with your own verified inputs before running. See docs/RASTER_INPUT.md for the complete input contract.")
    st.markdown(
        "- **Inputs**: precomputed feature rasters, genuine source coverage and a polygon boundary.\n"
        "- **Prediction extent**: the boundary and sufficient valid inputs; no OSAVI output mask is added.\n"
        "- **Outputs**: numeric GeoTIFFs, visualization files and quality reports.\n"
        "- **Interpretation**: plot trained models require independent assessment when applied at a new spatial scale or trial."
    )
    st.subheader("Feature extraction and training")
    st.code(
        "python -m wheatpheno extract --input examples/synthetic/feature_input.npz "
        "--output outputs/features --profile standard --block-size 10\n"
        "python -m wheatpheno evaluate --config configs/training.example.json --output outputs/training",
        language="bash",
    )
    st.caption("These commands use synthetic demonstration inputs. Evaluation of a new agricultural dataset requires its own measured outcomes and grouping information.")
    st.info(
        "Input names, units and formulas follow models/schema.json. "
        "NDRE = (NIR - RE) / (NIR + RE)."
    )


def page_contract(api: dict[str, Any], manifest: dict[str, Any]) -> None:
    st.title("Model Specification")
    st.caption("Feature definitions, units and model provenance for reproducible inference.")
    st.markdown(
        "**Fixed regression models** predict AGB and grain yield using their saved feature selection and preprocessing. "
        "Plant height is estimated separately from a canopy height percentile."
    )
    tabs = st.tabs(["AGB inputs", "Yield inputs", "PH inputs", "Model manifest"])
    for tab, target in zip(tabs[:3], TARGET_LABELS):
        with tab:
            names = required_features(api, target)
            st.dataframe(pd.DataFrame({"Order": range(1, len(names) + 1), "Feature": names}), width="stretch", hide_index=True)
            if target == "PH":
                st.caption("Table prediction accepts CHM_PCT95 in metres and returns PH in centimetres. No PH field maps or spatial examples are distributed.")
            else:
                st.caption("These are the required input columns. The stored model determines retained features and their order. Refer to the feature dictionary for formulas and units.")
    with tabs[3]:
        st.json(safe_display(manifest))
        st.download_button("Download model manifest", json_bytes(manifest), "model_manifest.json", "application/json")
    st.subheader("Compatibility requirements")
    st.markdown(
        "- **Spectral and texture features**: retain the training band order, reflectance scale, index formulas and GLCM settings.\n"
        "- **Fixed model inputs**: follow models/schema.json.\n"
        "- **NDRE**: (NIR - RE) / (NIR + RE).\n"
        "- **Structure**: use consistent ground references, elevation units and height sampling areas.\n"
        "- **Mapping**: no OSAVI threshold mask is added. Missing inputs and optional display filling are recorded separately."
    )


def page_quality(manifest: dict[str, Any]) -> None:
    st.title("Quality Reports")
    st.caption("Review input coverage, model provenance and outputs from this session.")
    runs = [("Table Prediction", st.session_state.get("csv_result"))]
    available = [(label, state) for label, state in runs if state]
    if not available:
        st.info("No table runs are available in this session. Reports appear here after table prediction; command line mapping reports are saved to the selected output directory.")
    for index, (label, state) in enumerate(available):
        st.subheader(label)
        root = Path(state["dir"])
        records = [
            {"File": path.relative_to(root).as_posix(), "Size (KB)": round(path.stat().st_size / 1024, 1)}
            for path in sorted(root.rglob("*")) if path.is_file()
        ]
        st.dataframe(pd.DataFrame(records), width="stretch", hide_index=True)
        display_reports(root, f"quality_{index}")
    with st.expander("Model provenance and integrity"):
        if "integrity_check" in st.session_state:
            st.json(safe_display(st.session_state["integrity_check"]))
        st.json(safe_display(manifest))
        st.caption("Integrity status comes from an executed model check. A manifest alone does not establish successful verification.")
    st.markdown(
        "**Interpretation**: run reports document inputs, models and valid coverage; they do not evaluate predictive accuracy. "
        "Study metrics use plot-grouped cross-validation. Public examples demonstrate execution."
    )


def main() -> None:
    # Remove results created before the revised input contract and publication
    # scope. In particular, an already open browser must not display withdrawn
    # field scenes or offer a cached archive for download after an app rerun.
    if st.session_state.get("wheatpheno_session_version") != SESSION_VERSION:
        for key in ("map_result", "map_archive", "csv_result", "integrity_check"):
            st.session_state.pop(key, None)
        st.session_state["wheatpheno_session_version"] = SESSION_VERSION
    for key in ("map_result", "map_archive"):
        st.session_state.pop(key, None)
    api = load_api()
    manifest = readable_manifest(api)
    with st.sidebar:
        st.markdown("## WheatPheno")
        st.caption("Open code and fixed models\n\nOptional local interface")
        page = st.radio("Workspace", ["Overview", "Table Prediction", "Spatial Mapping", "Model Specification", "Quality Reports"], label_visibility="collapsed")
        st.divider()
        st.markdown("**PH · AGB · Grain yield**")
        st.caption("21 table examples · Synthetic fixtures\n\nField mapping through the command line")
    if page == "Overview":
        page_overview(api, manifest)
    elif page == "Table Prediction":
        page_csv(api)
    elif page == "Spatial Mapping":
        page_mapping(api)
    elif page == "Model Specification":
        page_contract(api, manifest)
    else:
        page_quality(manifest)


if __name__ == "__main__":
    main()
