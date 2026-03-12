"""
manager — pacote de gestão de experimentos FG2P.

API pública:
    from manager import ExperimentManager
    m = ExperimentManager()
    m.show_missing()
    m.process_all_pending()
    m.rebuild_registry()
"""
from ._core import ExperimentManager

__all__ = ["ExperimentManager"]
