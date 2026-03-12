"""
Interface de linha de comando do manager.

Uso:
    python src/manage_experiments.py [flags]
    python -m src.manager [flags]

Exemplos:
    manage.py                         Lista todos os experimentos
    manage.py --show 13               Detalhes do experimento 13 (artefatos + gaps)
    manage.py --missing               Tabela de gaps (o que falta)
    manage.py --run 13                Roda pipeline pendente do exp 13
    manage.py --run 13 --force        Re-roda TUDO do exp 13
    manage.py --run                   Roda pipeline de TODOS
    manage.py --check                 Verifica consistencia (publicavel?)
    manage.py --clean 14              Apaga exp 14 (pede confirmacao)
    manage.py --clean-broken          Apaga todos incompletos/orfaos
    manage.py --compare exp0_baseline_70split
"""
import argparse
from ._core import ExperimentManager


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Gerenciador de Experimentos FG2P",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    # ----- Discovery -----
    parser.add_argument("--list",    action="store_true",
                        help="Lista experimentos (default sem flags)")
    parser.add_argument("--show",    type=int, metavar="N",
                        help="Detalhes do experimento N")
    parser.add_argument("--missing", action="store_true",
                        help="Tabela de gaps (eval/error_analysis/plot)")

    # ----- Pipeline -----
    parser.add_argument("--run",     nargs="?", const=-1, type=int, metavar="N",
                        help="Roda pipeline (N=indice, sem N=todos)")
    parser.add_argument("--force",   action="store_true",
                        help="Com --run: re-roda TODAS as etapas (inference incluso)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Simula sem executar")

    # ----- Cleanup -----
    parser.add_argument("--clean",         type=int, metavar="N",
                        help="Apaga experimento N (pede confirmacao)")
    parser.add_argument("--clean-broken",  action="store_true",
                        help="Apaga todos incompletos/orfaos (pede confirmacao)")

    # ----- Training -----
    parser.add_argument("--train",   type=str, metavar="CONFIG",
                        help="Treina conf/CONFIG.json e roda pipeline apos")

    # ----- Verificação e Compare -----
    parser.add_argument("--check",   action="store_true",
                        help="Verifica consistencia disco/performance.json (publicavel?)")
    parser.add_argument("--compare", type=str, metavar="EXP_NAME",
                        help="Compara runs do mesmo experimento")
    parser.add_argument("--registry", action="store_true",
                        help="Reconstroi model_registry.json")

    return parser


def main():
    parser  = build_parser()
    args    = parser.parse_args()
    manager = ExperimentManager()

    # --- Pipeline ---
    if args.run is not None:
        index = args.run if args.run != -1 else None
        manager.process_all_pending(
            dry_run=args.dry_run,
            force=args.force,
            force_inference=args.force,   # --force faz tudo
            index=index,
        )

    # --- Cleanup ---
    elif args.clean is not None:
        manager.prune_experiment(args.clean, dry_run=args.dry_run)
    elif args.clean_broken:
        manager.prune_incomplete(dry_run=args.dry_run)

    # --- Discovery ---
    elif args.show is not None:
        manager.show_experiment(args.show)
    elif args.missing:
        manager.show_missing()

    # --- Verificação ---
    elif args.check:
        manager.check_consistency()

    # --- Compare & Sync ---
    elif args.compare:
        manager._require_index_map()
        manager.compare_experiment(args.compare)
    elif args.registry:
        manager.rebuild_registry()

    # --- Training ---
    elif args.train:
        manager.train_experiment(args.train, dry_run=args.dry_run)

    # --- Default: list ---
    else:
        manager.list_experiments()
