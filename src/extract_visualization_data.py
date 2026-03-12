#!/usr/bin/env python3
"""
FG2P Visualization Data Extraction Utility

Extracts metrics, class distributions, and training info from:
- error_analysis_*.txt files
- metadata.json files
- config_*.json files
- model_registry.json

Provides both raw data access and high-level aggregation functions.
"""

import json
import re
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, asdict
import sys


@dataclass
class ExperimentMetrics:
    """Metrics extracted from error_analysis file"""
    exp_name: str
    run_id: str
    per: float
    wer: float
    accuracy: float
    total_words: int
    correct_words: int
    errors: int
    class_a_count: int
    class_a_pct: float
    class_b_count: int
    class_b_pct: float
    class_c_count: int
    class_c_pct: float
    class_d_count: int
    class_d_pct: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ModelMetadata:
    """Metadata extracted from metadata.json"""
    experiment_name: str
    run_id: str
    total_params: int
    hidden_dim: int
    emb_dim: int
    final_epoch: int
    total_time_seconds: float
    loss_type: str
    loss_lambda: Optional[float]
    uses_separators: bool
    train_size: int
    test_size: int
    val_size: int

    @property
    def total_params_m(self) -> float:
        """Parameters in millions"""
        return self.total_params / 1e6

    @property
    def total_time_hours(self) -> float:
        """Training time in hours"""
        return self.total_time_seconds / 3600

    @property
    def throughput_words_per_sec(self) -> float:
        """Approximate throughput (words/second)"""
        if self.total_time_seconds <= 0:
            return 0
        total_processed = self.train_size * self.final_epoch
        return total_processed / self.total_time_seconds

    def to_dict(self) -> dict:
        d = asdict(self)
        d['total_params_m'] = self.total_params_m
        d['total_time_hours'] = self.total_time_hours
        d['throughput_words_per_sec'] = self.throughput_words_per_sec
        return d


class ExperimentDataExtractor:
    """Extract metrics and metadata from FG2P results"""

    def __init__(self, root_dir: str = "."):
        self.root = Path(root_dir)
        self.results_dir = self.root / "results"
        self.models_dir = self.root / "models"
        self.conf_dir = self.root / "conf"

    def extract_metrics_from_error_analysis(
        self, error_file: Path
    ) -> Optional[ExperimentMetrics]:
        """Extract metrics from error_analysis_*.txt file"""
        try:
            with open(error_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # Extract experiment name and run_id from filename
            match = re.search(r"error_analysis_(.+?)__(\d{8}_\d{6})", error_file.name)
            if not match:
                return None
            exp_name, run_id = match.groups()

            # Extract metrics
            metrics = {}
            for metric_name, pattern in [
                ("per", r"PER:\s+([\d.]+)%"),
                ("wer", r"WER:\s+([\d.]+)%"),
                ("accuracy", r"Accuracy:\s+([\d.]+)%"),
            ]:
                m = re.search(pattern, content)
                if m:
                    metrics[metric_name] = float(m.group(1))

            # Extract total and correct
            total_match = re.search(
                r"Total: (\d+) palavras \| Corretas: (\d+) \| Erros: (\d+)",
                content
            )
            if total_match:
                metrics["total_words"] = int(total_match.group(1))
                metrics["correct_words"] = int(total_match.group(2))
                metrics["errors"] = int(total_match.group(3))

            # Extract class distribution
            # Format: "Classe A: 351656 (99.08%)" with variable spacing
            # Needs to handle: "Classe A: 351656 (99.08%)" and "Classe B:   1042 ( 0.29%)"
            class_pattern = r"Classe\s+([A-D]):\s+(\d+)\s+\(\s*([0-9.]+)%\)"
            class_matches = re.findall(class_pattern, content)
            for classe, count, pct in class_matches:
                classe_lower = classe.lower()  # Convert to lowercase
                metrics[f"class_{classe_lower}_count"] = int(count)
                metrics[f"class_{classe_lower}_pct"] = float(pct)

            # If no classes found, check without word-level (DISTRIBUIÇÃO DE CLASSES line)
            # Some old files may not have this section
            if not class_matches:
                # Older files may lack class distribution data
                for classe in ['A', 'B', 'C', 'D']:
                    metrics[f"class_{classe}_count"] = 0
                    metrics[f"class_{classe}_pct"] = 0.0

            # Validate we have all required fields
            required = {
                "per", "wer", "accuracy", "total_words", "correct_words", "errors",
                "class_a_count", "class_a_pct", "class_b_count", "class_b_pct",
                "class_c_count", "class_c_pct", "class_d_count", "class_d_pct"
            }
            if not required.issubset(set(metrics.keys())):
                missing = required - set(metrics.keys())
                print(f"Warning: Missing fields in {error_file.name}: {missing}")
                return None

            return ExperimentMetrics(
                exp_name=exp_name,
                run_id=run_id,
                per=metrics["per"],
                wer=metrics["wer"],
                accuracy=metrics["accuracy"],
                total_words=metrics["total_words"],
                correct_words=metrics["correct_words"],
                errors=metrics["errors"],
                class_a_count=metrics["class_a_count"],
                class_a_pct=metrics["class_a_pct"],
                class_b_count=metrics["class_b_count"],
                class_b_pct=metrics["class_b_pct"],
                class_c_count=metrics["class_c_count"],
                class_c_pct=metrics["class_c_pct"],
                class_d_count=metrics["class_d_count"],
                class_d_pct=metrics["class_d_pct"],
            )
        except Exception as e:
            print(f"Error extracting metrics from {error_file}: {e}")
            return None

    def extract_metadata(self, metadata_file: Path) -> Optional[ModelMetadata]:
        """Extract metadata from metadata.json file"""
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            loss_lambda = None
            if data.get("loss_config", {}).get("distance_lambda"):
                loss_lambda = data["loss_config"]["distance_lambda"]

            # Handle optional keep_syllable_separators field (may not exist in older files)
            uses_separators = data.get("config", {}).get("data", {}).get("keep_syllable_separators", False)

            return ModelMetadata(
                experiment_name=data["experiment_name"],
                run_id=data["run_id"],
                total_params=data["total_params"],
                hidden_dim=data["config"]["model"]["hidden_dim"],
                emb_dim=data["config"]["model"]["emb_dim"],
                final_epoch=data["final_epoch"],
                total_time_seconds=data["total_time_seconds"],
                loss_type=data.get("loss_type", "cross_entropy"),
                loss_lambda=loss_lambda,
                uses_separators=uses_separators,
                train_size=data["dataset"]["train_size"],
                val_size=data["dataset"]["val_size"],
                test_size=data["dataset"]["test_size"],
            )
        except Exception as e:
            # Skip silently for older files - just log at verbose level
            return None

    def discover_all_experiments(self) -> List[Dict]:
        """Discover all experiments with error_analysis files"""
        experiments = []

        if not self.results_dir.exists():
            print(f"Results directory not found: {self.results_dir}")
            return experiments

        for error_file in sorted(self.results_dir.glob("*/error_analysis_*.txt")):
            match = re.search(r"error_analysis_(.+?)__(\d{8}_\d{6})", error_file.name)
            if match:
                exp_name, run_id = match.groups()
                experiments.append({
                    "exp_dir": error_file.parent.name,
                    "exp_name": exp_name,
                    "run_id": run_id,
                    "error_analysis_path": error_file,
                })

        return experiments

    def load_all_metrics(self) -> Dict[str, ExperimentMetrics]:
        """Load metrics for all experiments"""
        all_metrics = {}
        experiments = self.discover_all_experiments()

        print(f"Loading metrics from {len(experiments)} experiments...")
        for exp in experiments:
            metrics = self.extract_metrics_from_error_analysis(exp["error_analysis_path"])
            if metrics:
                all_metrics[exp["exp_name"]] = metrics
                # Avoid Unicode issues on Windows by using simple output
                print(f"  OK {exp['exp_name']}: PER={metrics.per:.2f}%, WER={metrics.wer:.2f}%")

        return all_metrics

    def load_all_metadata(self) -> Dict[str, ModelMetadata]:
        """Load metadata for all experiments with saved models"""
        all_metadata = {}

        if not self.models_dir.exists():
            print(f"Models directory not found: {self.models_dir}")
            return all_metadata

        for metadata_file in sorted(self.models_dir.glob("*/*_metadata.json")):
            metadata = self.extract_metadata(metadata_file)
            if metadata:
                all_metadata[metadata.experiment_name] = metadata
                print(f"  OK {metadata.experiment_name}: {metadata.total_params_m:.1f}M params")

        return all_metadata

    def load_model_registry(self) -> Optional[Dict]:
        """Load model_registry.json for SOTA model aliases"""
        registry_file = self.models_dir / "model_registry.json"
        if not registry_file.exists():
            return None

        try:
            with open(registry_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading model_registry.json: {e}")
            return None

    def get_sota_models(self) -> Dict[str, Dict]:
        """Get SOTA model information from registry"""
        registry = self.load_model_registry()
        if not registry:
            return {}

        return registry.get("aliases", {})

    def combine_metrics_and_metadata(
        self,
        all_metrics: Dict[str, ExperimentMetrics],
        all_metadata: Dict[str, ModelMetadata],
    ) -> Dict[str, Dict]:
        """Combine metrics and metadata into single records"""
        combined = {}

        for exp_name, metrics in all_metrics.items():
            combined[exp_name] = {
                "metrics": metrics.to_dict(),
                "metadata": all_metadata[exp_name].to_dict() if exp_name in all_metadata else None,
            }

        return combined

    def top_by_metric(
        self,
        all_metrics: Dict[str, ExperimentMetrics],
        metric: str = "per",
        top_n: int = 5,
    ) -> List[Tuple[str, float]]:
        """Get top N experiments by metric"""
        sorted_exps = sorted(
            all_metrics.items(),
            key=lambda x: getattr(x[1], metric)
        )
        return sorted_exps[:top_n]

    def export_to_csv(self, all_metrics: Dict[str, ExperimentMetrics], output_file: str):
        """Export metrics to CSV for spreadsheet analysis"""
        import csv

        if not all_metrics:
            print("No metrics to export")
            return

        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=all_metrics[next(iter(all_metrics))].to_dict().keys())
            writer.writeheader()
            for metrics in all_metrics.values():
                writer.writerow(metrics.to_dict())

        print(f"Exported metrics to {output_file}")

    def export_to_json(self, combined: Dict[str, Dict], output_file: str):
        """Export combined data to JSON"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(combined, f, indent=2)

        print(f"Exported combined data to {output_file}")


def main():
    """CLI interface for data extraction"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract and aggregate FG2P visualization data"
    )
    parser.add_argument(
        "--root",
        default=".",
        help="FG2P project root directory"
    )
    parser.add_argument(
        "--metrics-csv",
        help="Export metrics to CSV file"
    )
    parser.add_argument(
        "--combined-json",
        help="Export combined metrics+metadata to JSON"
    )
    parser.add_argument(
        "--top-per",
        type=int,
        default=5,
        help="Show top N experiments by PER"
    )
    parser.add_argument(
        "--top-wer",
        type=int,
        default=5,
        help="Show top N experiments by WER"
    )
    parser.add_argument(
        "--sota",
        action="store_true",
        help="Show SOTA models from registry"
    )

    args = parser.parse_args()

    extractor = ExperimentDataExtractor(args.root)

    # Load all data
    print("\n=== Loading Metrics ===")
    all_metrics = extractor.load_all_metrics()
    print(f"Loaded metrics for {len(all_metrics)} experiments\n")

    print("=== Loading Metadata ===")
    all_metadata = extractor.load_all_metadata()
    print(f"Loaded metadata for {len(all_metadata)} models\n")

    # Show SOTA models
    if args.sota:
        print("=== SOTA Models ===")
        sota = extractor.get_sota_models()
        for alias, info in sota.items():
            print(f"{alias}:")
            print(f"  Experiment: {info.get('experiment')}")
            print(f"  PER: {info.get('per', 'N/A'):.4f}")
            print(f"  WER: {info.get('wer', 'N/A'):.4f}")
            print()

    # Show top by PER
    print(f"=== Top {args.top_per} by PER ===")
    for exp_name, metrics in extractor.top_by_metric(all_metrics, "per", args.top_per):
        print(f"{exp_name}: PER={metrics.per:.2f}%, WER={metrics.wer:.2f}%, "
              f"Acc={metrics.accuracy:.2f}%")

    # Show top by WER
    print(f"\n=== Top {args.top_wer} by WER ===")
    for exp_name, metrics in extractor.top_by_metric(all_metrics, "wer", args.top_wer):
        print(f"{exp_name}: PER={metrics.per:.2f}%, WER={metrics.wer:.2f}%, "
              f"Acc={metrics.accuracy:.2f}%")

    # Export to files
    if args.metrics_csv:
        extractor.export_to_csv(all_metrics, args.metrics_csv)

    if args.combined_json:
        combined = extractor.combine_metrics_and_metadata(all_metrics, all_metadata)
        extractor.export_to_json(combined, args.combined_json)


if __name__ == "__main__":
    main()
