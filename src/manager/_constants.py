"""Caminhos e constantes compartilhadas entre todos os submódulos do manager."""
from pathlib import Path
import sys

# Fix de encoding para console Windows
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

_SRC_DIR = Path(__file__).resolve().parent.parent        # src/
_ROOT    = _SRC_DIR.parent                               # FG2P/

if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from utils import MODELS_DIR, RESULTS_DIR, get_logger  # noqa: E402

PERFORMANCE_PATH = _ROOT / "docs" / "report" / "performance.json"
REGISTRY_PATH    = MODELS_DIR / "model_registry.json"
REPORT_PATH      = RESULTS_DIR / "model_report.html"

logger = get_logger("manager")
