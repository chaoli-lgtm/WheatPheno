from pathlib import Path
import sys

def project_root():
    source = Path(__file__).resolve().parents[2]
    if (source / "examples").is_dir():
        return source
    installed = Path(sys.prefix) / "share" / "wheatpheno"
    if (installed / "examples").is_dir():
        return installed
    raise FileNotFoundError("WheatPheno example assets are not installed; use the source release.")
