import logging
from pathlib import Path
from datetime import datetime

# PATHS
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
RESULTS_DIR = PROJECT_ROOT / "results"

# Create dirs if not exist
for d in [DATA_DIR, MODELS_DIR, LOGS_DIR, RESULTS_DIR]:
    d.mkdir(exist_ok=True)

# LOGGING
def get_logger(name):
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '[%(asctime)s] %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger

# FILE NAMES
def get_model_path(name="g2p_model"):
    return MODELS_DIR / f"{name}.pt"

def get_log_path(name="training"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return LOGS_DIR / f"{name}_{timestamp}.log"

def get_result_path(name="results"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return RESULTS_DIR / f"{name}_{timestamp}.txt"


# ═══════════════════════════════════════════════════════════════════════════
# PRÉ-PROCESSAMENTO DE PALAVRAS
# ═══════════════════════════════════════════════════════════════════════════

# Mapeamento de caracteres não-treinados para similares treinados
# Baseado em análise de dicts/pt-br.tsv
# Caracteres treinados (lowercased): a á à â ã b c ç d e é ê f g h i í j l m n o ó ô õ p q r s t u ú ü v x z -
# Ver: docs/06_PREPROCESSING.md para teoria completa
CHAR_MAPPING = {
    # ─────────────────────────────────────────────────────────────────────────
    # Variações de acentos não-treinadas → versões treinadas mais próximas
    # ─────────────────────────────────────────────────────────────────────────

    # Variantes de 'a'
    'ā': 'a',      # macron (linguística)
    'ă': 'a',      # breve
    'ą': 'a',      # ogonek
    'å': 'a',      # ring (escandinavo: Stockholm → estocolmo)

    # Variantes de 'e'
    'ē': 'e',      # macron
    'ĕ': 'e',      # breve
    'ė': 'e',      # dot above
    'ę': 'e',      # ogonek
    'ë': 'e',      # diaeresis (francês: Noël → noel)

    # Variantes de 'i'
    'ī': 'i',      # macron
    'ĭ': 'i',      # breve
    'ï': 'i',      # diaeresis (francês: naïve → naive)
    'ı': 'i',      # dotless i (turco)

    # Variantes de 'o'
    'ō': 'o',      # macron
    'ŏ': 'o',      # breve
    'ö': 'o',      # diaeresis (alemão: München → munique)
    'ø': 'o',      # stroke (nórdico)
    'œ': 'o',      # ligadura ae → o (aproximação)

    # Variantes de 'u'
    'ū': 'u',      # macron
    'ŭ': 'u',      # breve
    'ů': 'u',      # ring (checo)
    'ų': 'u',      # ogonek

    # Consoantes especiais
    'ñ': 'nh',     # espanhol ñ → português nh (mañana → manha)
    'š': 's',      # háček (eslavo)
    'č': 'c',      # háček (checo, croata)
    'ž': 'z',      # háček (eslavo)
    'đ': 'd',      # stroke (sérvio, croata)
    'ð': 'd',      # eth (islandês)
    'þ': 't',      # thorn (islandês)
    'ł': 'l',      # stroke (polonês)
    'ß': 's',      # eszett (alemão)

    # Outros
    'ý': 'i',      # y + agudo → i (mais natural em PT-BR)
    'ŷ': 'i',      # y + circunflexo
    'ÿ': 'i',      # y + diaeresis
}

def preprocess_word(word: str, normalize: bool = True, warn_changes: bool = False) -> tuple[str, dict]:
    """
    Pré-processa uma palavra para inferência G2P.

    Normaliza caracteres não-treinados para versões treinadas.
    Útil para palavras com maiúsculas, acentos especiais, etc.

    Exemplos:
        "Computador"    → ("computador", {"original": "Computador", "modified": False})
        "Lazzaretti"    → ("lazzaretti", {"original": "Lazzaretti", "modified": False})
        "naïve" (ï)     → ("naive", {"original": "naïve", "modified": True, "changes": {"ï": "i"}})

    Args:
        word: Palavra original
        normalize: Se True, aplica normalização
        warn_changes: Se True, retorna info sobre mudanças feitas

    Returns:
        (palavra_processada, info_dict)
        info_dict contém:
            "original": palavra original
            "modified": True se houve mudanças
            "changes": dict com mapeamentos feitos (se modified=True)
    """
    info = {
        "original": word,
        "modified": False,
        "changes": {}
    }

    if not normalize:
        return word, info

    processed = word.lower()

    # Aplicar mapeamento de caracteres
    for old_char, new_char in CHAR_MAPPING.items():
        if old_char in processed:
            processed = processed.replace(old_char, new_char)
            info["changes"][old_char] = new_char
            info["modified"] = True

    if word.lower() != processed:
        info["modified"] = True

    return processed, info


def validate_word_charset(word: str, valid_chars: set) -> tuple[bool, list]:
    """
    Valida se todos os caracteres da palavra estão no charset treinado.

    Args:
        word: Palavra a validar
        valid_chars: Set de caracteres válidos (ex: corpus.char_vocab.vocab)

    Returns:
        (is_valid, invalid_chars_list)
    """
    invalid = [c for c in word if c not in valid_chars]
    return len(invalid) == 0, invalid


# ═══════════════════════════════════════════════════════════════════════════


def _count_lines(path: Path, skip_comments: bool = True) -> int:
    """Conta linhas de um arquivo, opcionalmente ignorando comentários.
    
    Args:
        path: Caminho do arquivo
        skip_comments: Se True, ignora linhas que começam com '#'
    
    Returns:
        Número de linhas (0 se arquivo não existir)
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            if skip_comments:
                return sum(1 for line in f if not line.startswith("#"))
            return sum(1 for _ in f)
    except FileNotFoundError:
        return 0


def get_cache_info(cache_dir: Path, train_name: str = "train.txt",
                   val_name: str = "val.txt",
                   test_name: str = "test.txt") -> dict:
    """Obtém informações sobre arquivos de cache de split.
    
    Args:
        cache_dir: Diretório contendo os arquivos de split
        train_name: Nome do arquivo de treino
        val_name: Nome do arquivo de validação
        test_name: Nome do arquivo de teste
    
    Returns:
        Dict com paths, contagens de linhas, e flags de existência
    """
    cache_dir = Path(cache_dir)
    train_path = cache_dir / train_name
    val_path = cache_dir / val_name
    test_path = cache_dir / test_name
    train_count = _count_lines(train_path)
    val_count = _count_lines(val_path)
    test_count = _count_lines(test_path)

    return {
        "dir": str(cache_dir),
        "train_path": str(train_path),
        "val_path": str(val_path),
        "test_path": str(test_path),
        "train_count": train_count,
        "val_count": val_count,
        "test_count": test_count,
        "train_exists": train_path.exists(),
        "val_exists": val_path.exists(),
        "test_exists": test_path.exists(),
    }


def log_dataset_summary(logger, meta: dict | None, cache_info: dict | None = None):
    logger.info("DATASET (resumo)")

    if not meta:
        logger.info("  Metadados do dataset indisponiveis")
        return

    dict_path = meta.get("dict_path", "N/A")
    checksum = meta.get("dict_checksum", "N/A")
    total_words = meta.get("total_words", 0)
    train_size = meta.get("train_size", 0)
    val_size = meta.get("val_size", 0)
    test_size = meta.get("test_size", 0)
    val_ratio = meta.get("val_ratio", 0.0)
    test_ratio = meta.get("test_ratio", 0.0)
    train_ratio = 1.0 - val_ratio - test_ratio
    seed = meta.get("seed", "?")
    char_vocab = meta.get("char_vocab_size", 0)
    phon_vocab = meta.get("phoneme_vocab_size", 0)
    strata_count = meta.get("strata_count", 0)

    logger.info("  Dicionario: %s", dict_path)
    logger.info("  Checksum: %s", checksum)
    logger.info(
        "  Total: %d | Train: %d | Val: %d | Test: %d | Split: %.0f%%/%.0f%%/%.0f%% | Seed: %s",
        total_words,
        train_size,
        val_size,
        test_size,
        train_ratio * 100,
        val_ratio * 100,
        test_ratio * 100,
        seed,
    )
    logger.info(
        "  Vocab: chars=%d | phonemes=%d | estratos=%d",
        char_vocab,
        phon_vocab,
        strata_count,
    )

    verdict = meta.get("verdict", {}) if isinstance(meta.get("verdict", {}), dict) else {}
    quality = verdict.get("quality", "N/A")
    confidence = verdict.get("confidence", "N/A")
    bonf_alpha = verdict.get("bonferroni_alpha", None)
    min_p = verdict.get("min_pvalue", None)
    max_cv = verdict.get("max_cramers_v", None)

    if bonf_alpha is not None and min_p is not None and max_cv is not None:
        logger.info(
            "  Split: %s | Confianca: %s | Bonferroni a=%.4f | min p=%.4e | max Cramer V=%.6f",
            quality,
            confidence,
            bonf_alpha,
            min_p,
            max_cv,
        )
    else:
        logger.info("  Split: %s | Confianca: %s", quality, confidence)

    train_ph_cov = meta.get("train_phoneme_coverage", None)
    train_bg_cov = meta.get("train_bigram_coverage", None)
    val_ph_cov = meta.get("val_phoneme_coverage", None)
    val_bg_cov = meta.get("val_bigram_coverage", None)
    if train_ph_cov is not None and train_bg_cov is not None:
        logger.info(
            "  Cobertura (treino): fonemas %.1f%% | bigramas %.1f%%",
            train_ph_cov * 100,
            train_bg_cov * 100,
        )
    if val_ph_cov is not None and val_bg_cov is not None:
        logger.info(
            "  Cobertura (val):    fonemas %.1f%% | bigramas %.1f%%",
            val_ph_cov * 100,
            val_bg_cov * 100,
        )

    if cache_info:
        if cache_info.get("train_exists") or cache_info.get("test_exists"):
            logger.info(
                "  Cache: %s | train=%d | val=%d | test=%d",
                cache_info.get("dir", "N/A"),
                cache_info.get("train_count", 0),
                cache_info.get("val_count", 0),
                cache_info.get("test_count", 0),
            )
        else:
            logger.info("  Cache: %s (nao encontrado)", cache_info.get("dir", "N/A"))


def log_common_header(logger, title: str, timestamp: str, device, dataset_meta: dict | None = None,
                      cache_info: dict | None = None):
    logger.info("")
    logger.info("=" * 70)
    logger.info("G2P LSTM - %s", title)
    logger.info("=" * 70)
    logger.info("Timestamp: %s", timestamp)
    logger.info("Device: %s", device)
    log_dataset_summary(logger, dataset_meta, cache_info)
    logger.info("=" * 70)
    logger.info("")


# ============================================================================
# Model and Artifact Management
# ============================================================================

def get_all_models_sorted() -> list[Path]:
    """Retorna lista de todos os modelos ordenados por tempo de modificação
    
    Esta função é a fonte única de verdade para ordenação de modelos.
    Todos os scripts (inference.py, manage_experiments.py, etc.) devem usar
    esta função para garantir indexação consistente.
    
    Returns:
        Lista de Paths ordenada por st_mtime (mais antigo → mais recente)
    """
    return sorted(MODELS_DIR.glob("*.pt"), key=lambda p: p.stat().st_mtime)


def find_latest_model() -> Path | None:
    """Encontra o modelo (.pt) mais recente em models/"""
    model_files = get_all_models_sorted()
    return model_files[-1] if model_files else None


def find_model_by_pattern(pattern: str) -> list[Path]:
    """Busca modelos que correspondem ao padrão
    
    Args:
        pattern: String a buscar no nome do modelo (case-insensitive)
        
    Returns:
        Lista de Paths dos arquivos .pt que correspondem
    """
    all_models = list(MODELS_DIR.glob("*.pt"))
    return [m for m in all_models if pattern.lower() in m.stem.lower()]


def get_model_name_from_file(model_path: Path) -> str:
    """Extrai o nome base do modelo (sem extensão)"""
    return model_path.stem


def list_model_artifacts(model_name: str) -> dict[str, Path | list[Path] | None]:
    """Lista todos os artefatos relacionados a um modelo
    
    Args:
        model_name: Nome base do modelo (sem extensão)
        
    Returns:
        Dict com paths de todos artefatos encontrados:
        - model_file: arquivo .pt
        - metadata_file: arquivo _metadata.json
        - history_csv: arquivo _history.csv
        - evaluation_files: lista de evaluation_*.txt que podem estar relacionados
        - prediction_files: lista de predictions_*.tsv
        - analysis_plots: lista de _analysis.png, _convergence.png, etc.
        - summary_file: arquivo _summary.txt
    """
    artifacts = {
        "model_file": MODELS_DIR / f"{model_name}.pt",
        "metadata_file": MODELS_DIR / f"{model_name}_metadata.json",
        "history_csv": RESULTS_DIR / f"{model_name}_history.csv",
        "summary_file": RESULTS_DIR / f"{model_name}_summary.txt",
        "evaluation_files": [],
        "prediction_files": [],
        "analysis_plots": [],
        "error_analysis_files": [],
    }
    
    # Verificar existência dos arquivos principais
    for key in ["model_file", "metadata_file", "history_csv", "summary_file"]:
        if not artifacts[key].exists():
            artifacts[key] = None
    
    # Buscar evaluation files (podem ter timestamp diferente)
    artifacts["evaluation_files"] = list(RESULTS_DIR.glob("evaluation_*.txt"))
    
    # Buscar prediction files
    artifacts["prediction_files"] = list(RESULTS_DIR.glob("predictions_*.tsv"))
    
    # Buscar error analysis files
    artifacts["error_analysis_files"] = list(RESULTS_DIR.glob("error_analysis_*.txt"))
    
    # Buscar plots específicos do modelo
    artifacts["analysis_plots"] = list(RESULTS_DIR.glob(f"{model_name}_*.png"))
    
    return artifacts


def get_orphan_files() -> dict[str, list[Path]]:
    """Identifica arquivos em results/ que não têm modelo correspondente
    
    Returns:
        Dict com listas de arquivos órfãos por tipo
    """
    # Listar todos os modelos existentes
    model_names = {m.stem for m in MODELS_DIR.glob("*.pt")}
    
    orphans = {
        "evaluation": [],
        "predictions": [],
        "error_analysis": [],
        "plots": [],
        "histories": [],
        "summaries": [],
    }
    
    # Evaluation files são difíceis de vincular (usam timestamp)
    # Consideramos órfãos se houver "muitos" (>10 sem predictions correspondentes)
    eval_files = list(RESULTS_DIR.glob("evaluation_*.txt"))
    pred_files = list(RESULTS_DIR.glob("predictions_*.tsv"))
    
    # Se houver muitos evaluations e poucos predictions, há órfãos
    if len(eval_files) > len(pred_files) * 3:
        # Considerar os mais antigos como órfãos (preservar últimos 5)
        orphans["evaluation"] = sorted(eval_files, key=lambda p: p.stat().st_mtime)[:-5]
    
    # Histories e summaries órfãos (sem modelo .pt correspondente)
    for hist in RESULTS_DIR.glob("*_history.csv"):
        model_name = hist.stem.replace("_history", "")
        if model_name not in model_names:
            orphans["histories"].append(hist)
    
    for summary in RESULTS_DIR.glob("*_summary.txt"):
        model_name = summary.stem.replace("_summary", "")
        if model_name not in model_names:
            orphans["summaries"].append(summary)
    
    # Plots órfãos
    for plot in RESULTS_DIR.glob("*.png"):
        # Extrair nome do modelo do plot (antes do primeiro _ após o nome)
        parts = plot.stem.split("_")
        # Tentar identificar padrões como 'g2p_exp2_..._20260216_XXXXX_analysis.png'
        # Se não conseguirmos vincular, consideramos órfão se for antigo
        found = False
        for model_name in model_names:
            if plot.stem.startswith(model_name):
                found = True
                break
        if not found:
            orphans["plots"].append(plot)
    
    return orphans


def get_model_summary(model_name: str) -> dict:
    """Retorna sumário completo de um modelo (metadata + artefatos)
    
    Args:
        model_name: Nome base do modelo
        
    Returns:
        Dict com:
        - name: nome do modelo
        - exists: se modelo .pt existe
        - metadata: dict com metadados (ou None)
        - artifacts: dict com artefatos encontrados (chaves normalizadas)
        - completed: se treinamento foi completado
        - counts: dict com contagens {training, evaluation, total}
    """
    import json
    
    artifacts_raw = list_model_artifacts(model_name)
    
    # Carregar metadata se existir
    metadata = None
    completed = False
    if artifacts_raw.get("metadata_file"):
        try:
            with open(artifacts_raw["metadata_file"], "r", encoding="utf-8") as f:
                metadata = json.load(f)
                completed = metadata.get("training_completed", False)
        except Exception:
            pass
    
    # Reorganizar artifacts para formato consistente (normalização de chaves)
    artifacts = {
        "model": artifacts_raw.get("model_file"),
        "metadata": artifacts_raw.get("metadata_file"),
        "history": artifacts_raw.get("history_csv"),
        "evaluations": artifacts_raw.get("evaluation_files", []),
        "predictions": artifacts_raw.get("prediction_files", []),
        "summaries": artifacts_raw.get("error_analysis_files", []),
        "plots": artifacts_raw.get("analysis_plots", []),
    }
    
    # Contar por categoria
    training_count = sum([
        1 if artifacts["history"] else 0,
        len(artifacts["plots"]),
    ])
    
    evaluation_count = sum([
        len(artifacts["evaluations"]),
        len(artifacts["predictions"]),
        len(artifacts["summaries"]),
    ])
    
    total_count = 2 + training_count + evaluation_count  # +2 for model+metadata
    
    return {
        "name": model_name,
        "exists": artifacts["model"] is not None,
        "metadata": metadata,
        "artifacts": artifacts,
        "completed": completed,
        "counts": {
            "training": training_count,
            "evaluation": evaluation_count,
            "total": total_count,
        },
    }
