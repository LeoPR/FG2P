#!/usr/bin/env python
"""
inference_light.py — Tutorial: como usar o FG2P para inferência G2P.

Este arquivo é um guia de uso, independente do restante do projeto.
Pode ser copiado e adaptado livremente.

═══ MÍNIMO (basta isso para inferir) ═══════════════════════════════════════
    from src.inference_light import G2PPredictor

    p = G2PPredictor.load("best_per")  # melhor PER — recomendado para TTS
    p = G2PPredictor.load("best_wer")  # melhor WER — recomendado para NLP
    p.predict("computador")            # → "k õ p u t a ˈ d o x"

═══ ALTA PERFORMANCE — modo batch (corpus, pipeline, pré-processamento) ════
    results = p.predict_batch_native(lista_de_palavras, batch_size=32)
    # CPU (Xeon): batch=32 → 155 w/s (+6.5× vs predict() em loop)
    #             batch=128 → 190 w/s (pico CPU; saturação em ~64)
    # GPU (RTX 3060): batch=32 → 406 w/s (+11.8×); batch=512 → 1.106 w/s (pico)
    # Recomendação: batch=32 para uso geral; batch=128 para ingestão máxima CPU;
    #               batch=512 para ingestão máxima GPU.
    # Para TTS palavra a palavra: predict() — latência p50 42ms (CPU) / 28ms (GPU)

═══ CLI (da linha de comando) ══════════════════════════════════════════════
    python src/inference_light.py --alias best_per --word computador
    python src/inference_light.py --index 11 --words "selfie,drone,blog"
    python src/inference_light.py --file corpus.txt --batch-size 128   # CPU pico
    python src/inference_light.py --file corpus.txt --batch-size 512   # GPU pico
    python src/inference_light.py --interactive
    python src/inference_light.py --list

═══ COMPARAÇÃO DE PALAVRAS SIMILARES — análise de famílias de palavras ═══════
    # Encontra palavras similares (ortograficamente) e compara pronúncias:
    python src/inference_light.py --word laeta --similar
    python src/inference_light.py --index 18 --word "computador" --similar --similar-count 10

═══ NEOLOGISMOS — análise rica de OOV e empréstimos ════════════════════════
    # Avalia banco de 35 neologismos com diff e notas linguísticas:
    p.evaluate_neologisms()
    p.evaluate_neologisms("docs/data/neologisms_test.tsv")  # path explícito

    python src/inference_light.py --neologisms
    python src/inference_light.py --index 18 --neologisms

═══ COM AVALIAÇÃO — opcional (se tiver referências) ════════════════════════
    # Carrega TSV genérico com fonemas de referência e calcula WER/PER:
    p.evaluate_tsv("docs/data/neologisms_test.tsv",
                   cache_tag="neologismos", cache_dir="data")

    python src/inference_light.py --tsv docs/data/neologisms_test.tsv \\
                                  --cache-tag neologismos

═══ AVALIAÇÃO COMPLETA (dataset de treino/teste padrão) ════════════════════
    python src/inference.py --index 11
"""

import argparse
import difflib
import sys
import torch
from pathlib import Path

# ---------------------------------------------------------------------------
# Tabela de grupos articulatórios PT-BR para métrica de aproximação fonológica.
# Independente do modelo G2P — pode ser usada como avaliador externo.
# ---------------------------------------------------------------------------
_PHONEME_GROUPS: dict = {
    # Vogais orais
    "i": "V_alto_front",   "ɪ": "V_alto_front",
    "u": "V_alto_back",    "ʊ": "V_alto_back",
    "e": "V_med_front",    "ɛ": "V_med_front",
    "o": "V_med_back",     "ɔ": "V_med_back",
    "ə": "V_central",      "a": "V_baixo",
    # Vogais nasais (PT-BR)
    "ã": "VN_baixo",       "ẽ": "VN_med_front",
    "ĩ": "VN_alto_front",  "ũ": "VN_alto_back",
    "õ": "VN_med_back",    "ʊ̃": "VN_alto_back",
    # Semivogais
    "y": "SV_palatal",     "w": "SV_labial",
    # Oclusivas
    "p": "OC_bilabial",    "b": "OC_bilabial",
    "t": "OC_dental",      "d": "OC_dental",
    "k": "OC_velar",       "ɡ": "OC_velar",
    # Fricativas
    "f": "FR_labio",       "v": "FR_labio",
    "s": "FR_dental",      "z": "FR_dental",
    "ʃ": "FR_palatal",     "ʒ": "FR_palatal",
    "x": "FR_velar",
    # Nasais
    "m": "NS_bilabial",    "n": "NS_dental",    "ɲ": "NS_palatal",
    # Laterais e vibrante
    "l": "LT_dental",      "ʎ": "LT_palatal",  "ɾ": "RHOT",
}


def _group_dist(p1: str, p2: str) -> float:
    """
    Distância articulatória aproximada entre dois fonemas PT-BR (0.0–1.0).

    Usa _PHONEME_GROUPS: mesmo grupo → 0.1 (ex: p↔b vozeamento, e↔ɛ altura);
    mesma classe major → 0.4; ambas vogais → 0.5; consoantes de classes
    diferentes → 0.55; vogal↔consoante → 0.9; estrutural ou desconhecido → 0.8.
    """
    if p1 == p2:
        return 0.0
    if p1 in {"ˈ", ".", "<EOS>", "<PAD>", "<UNK>"} or \
       p2 in {"ˈ", ".", "<EOS>", "<PAD>", "<UNK>"}:
        return 1.0
    g1 = _PHONEME_GROUPS.get(p1, "")
    g2 = _PHONEME_GROUPS.get(p2, "")
    if not g1 or not g2:
        return 0.8
    if g1 == g2:
        return 0.1
    pre1, pre2 = g1.split("_")[0], g2.split("_")[0]
    if pre1 == pre2:
        return 0.4
    if pre1.startswith("V") and pre2.startswith("V"):
        return 0.5
    if not pre1.startswith("V") and not pre2.startswith("V"):
        return 0.55
    return 0.9

# Windows: terminal cp1252 não suporta IPA — forçar UTF-8
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Setup de paths (necessário apenas ao executar diretamente como script)
# ---------------------------------------------------------------------------
_SRC_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SRC_DIR.parent

if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

# ---------------------------------------------------------------------------
# Imports das classes do projeto
# ---------------------------------------------------------------------------
from utils import get_logger, MODELS_DIR  # noqa: E402
from g2p import G2PCorpus, G2PLSTMModel, transform_grapheme_word  # noqa: E402
from inference import (  # noqa: E402
    predict_word, load_model_metadata, _get_complete_model_files,
)
from analyze_errors import (  # noqa: E402
    calculate_wer, calculate_per, calculate_accuracy,
)

DICT_PATH = _PROJECT_ROOT / "dicts" / "pt-br.tsv"
logger = get_logger("inference_light")


# =============================================================================
# PARTE 1 — PREDIÇÃO (mínimo necessário)
#
# Para inferir uma palavra, são necessários apenas:
#   1. Carregar o modelo (G2PPredictor.load)
#   2. Chamar predict(palavra)
# =============================================================================

class G2PPredictor:
    """
    Preditor G2P: palavra grafêmica → string de fonemas IPA.

    Replica o pipeline de treino para garantir consistência:
      G2PCorpus → transform_grapheme_word → char_vocab → model → phoneme_vocab

    Uso mínimo:
        p = G2PPredictor.load()
        p.predict("computador")     # "k õ p u t a ˈ d o x"
    """

    def __init__(self, model: G2PLSTMModel, corpus: G2PCorpus,
                 metadata: dict, device: torch.device):
        self.model    = model
        self.corpus   = corpus    # contém char_vocab, phoneme_vocab, grapheme_encoding
        self.metadata = metadata
        self.device   = device
        self.model.eval()

    # -------------------------------------------------------------------------
    # Carregamento do modelo
    # -------------------------------------------------------------------------

    @classmethod
    def load(cls, alias: str = None, model_path: Path = None, index: int = None,
             dict_path: Path = None, device: torch.device = None,
             quantize: bool = False, num_threads: int = None) -> "G2PPredictor":
        """
        Carrega modelo treinado. Ponto de entrada principal.

        Args:
            alias:       Nome semântico do modelo: "best_per", "best_wer", "fast".
                         Recomendado para uso externo — resolve via models/model_registry.json.
            index:       Índice na lista de modelos (0-based). None → mais recente.
            model_path:  Path direto para .pt (alternativa ao index/alias).
            dict_path:   Dicionário TSV. Padrão: dicts/pt-br.tsv.
            device:      CPU ou CUDA. Detectado automaticamente se None.
            quantize:    INT8 dynamic quantization (CPU only). Padrão: False.
                         Testado em Xeon+Windows: regressão de ~38% (dequantização por passo
                         domina para LSTM pequeno). Pode ganhar em ARM (Apple Silicon, mobile)
                         ou modelos maiores (hidden≥1024). Use --quantize para testar no seu
                         hardware — ver docs/evaluations/answered/016.
            num_threads: Threads intra-op do PyTorch para CPU (padrão: sistema/MKL).
                         Para inferência unitária (batch=1), MKL multi-thread já é ótimo —
                         não ajuste sem validar no seu hardware (Xeon: threads=1 causou
                         regressão de 61%). Use None para manter o padrão do sistema.

        Exemplos:
            p = G2PPredictor.load("best_per")              # melhor PER (recomendado TTS)
            p = G2PPredictor.load("best_wer")              # melhor WER (recomendado NLP)
            p = G2PPredictor.load()                        # modelo mais recente
            p = G2PPredictor.load(index=11)                # modelo por índice (ver --list)
            p = G2PPredictor.load("best_per", quantize=True)   # testar INT8 no seu hardware
            p = G2PPredictor.load("best_per", num_threads=4)   # mais threads (batch/throughput)
        """
        global DICT_PATH
        if dict_path is not None:
            DICT_PATH = Path(dict_path)

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Resolver qual modelo usar
        if alias is not None:
            model_path = cls._resolve_alias(alias)
            logger.info(f"Alias '{alias}' → {model_path.name}")
        elif model_path is None:
            models = _get_complete_model_files()
            if not models:
                raise RuntimeError("Nenhum modelo completo encontrado em models/")
            idx = len(models) - 1 if index is None else index
            if not (0 <= idx < len(models)):
                raise IndexError(
                    f"Índice {index} inválido (disponíveis: 0–{len(models)-1}). "
                    "Use G2PPredictor.list_models() para ver."
                )
            model_path = models[idx]
            logger.info(f"Modelo [{idx}]: {model_path.name}")
        else:
            model_path = Path(model_path)

        # Ler configuração salva com o modelo
        metadata = load_model_metadata(model_path)
        model_cfg, data_cfg = {}, {}
        if metadata and "config" in metadata:
            model_cfg = metadata["config"].get("model", {})
            data_cfg  = metadata["config"].get("data", metadata["config"])

        grapheme_encoding = data_cfg.get("grapheme_encoding", "raw")
        # grapheme_encoding pode ser str ("raw"/"decomposed") ou dict ({"type": "raw", "filters": [...]})
        # G2PCorpus.__init__ aceita ambos
        keep_sep          = bool(data_cfg.get("keep_syllable_separators", False))

        # Carregar corpus — constrói char_vocab e phoneme_vocab
        # (idêntico ao que train.py faz)
        logger.info(f"Carregando corpus ({DICT_PATH.name}) ...")
        corpus = G2PCorpus(DICT_PATH,
                           grapheme_encoding=grapheme_encoding,
                           keep_syllable_separators=keep_sep)

        # Instanciar modelo com a mesma arquitetura do treino
        emb_dim    = int(model_cfg.get("emb_dim", 128))
        hidden_dim = int(model_cfg.get("hidden_dim", 256))
        num_layers = int(model_cfg.get("num_layers", 2))
        dropout    = float(model_cfg.get("dropout", 0.5))
        emb_type   = str(model_cfg.get("embedding_type", "learned"))

        extra = {"phoneme_i2p": corpus.phoneme_vocab.i2p} if emb_type == "panphon" else {}
        model = G2PLSTMModel(
            len(corpus.char_vocab), len(corpus.phoneme_vocab),
            emb_dim=emb_dim, hidden_dim=hidden_dim,
            num_layers=num_layers, dropout=dropout,
            embedding_type=emb_type, **extra,
        ).to(device)

        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()

        # INT8 dynamic quantization — CPU only, zero training cost.
        # Padrão desativado: em Xeon+Windows causou regressão de 38% (dequantização
        # por passo autoregressivo domina para LSTM hidden=384). Pode ser vantajoso
        # em ARM (Apple Silicon, mobile) ou modelos com hidden≥1024. Ver doc 016.
        if device.type == "cpu" and quantize:
            import torch.nn as nn
            model = torch.quantization.quantize_dynamic(
                model, {nn.LSTM, nn.Linear}, dtype=torch.qint8
            )
            logger.info("INT8 dynamic quantization applied (CPU)")

        # Threads intra-op — CPU only.
        # Para inferência unitária sequencial (batch=1, 50 passos autoreg.), operações
        # são pequenas (~384-dim matmul). Com múltiplos threads, o overhead de
        # sincronização entre threads supera o ganho de paralelismo.
        # num_threads=1 elimina esse overhead; num_threads=None mantém padrão do sistema.
        if device.type == "cpu" and num_threads is not None:
            torch.set_num_threads(num_threads)
            logger.info(f"CPU intra-op threads: {num_threads}")

        n = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(
            f"{metadata.get('experiment_name', model_path.stem)} | "
            f"{n:,}p | enc={grapheme_encoding} sep={'S' if keep_sep else 'N'}"
        )
        return cls(model, corpus, metadata, device)

    # -------------------------------------------------------------------------
    # Predição — o núcleo do preditor
    # -------------------------------------------------------------------------

    def predict(self, word: str) -> str:
        """
        Palavra → fonemas.

        Pipeline (idêntico ao treino):
          1. transform_grapheme_word  — normaliza grafemas (no-op em modo 'raw')
          2. char_vocab.encode        — converte letras em índices
          3. model.predict            — decoder autoregressivo LSTM
          4. phoneme_vocab.decode     — converte índices em símbolos IPA

        Args:
            word: Palavra grafêmica, ex: "computador".

        Returns:
            Fonemas separados por espaço, ex: "k õ p u t a ˈ d o x".
            Modelos com separadores silábicos incluem '.' na saída.
        """
        transformed = transform_grapheme_word(word, self.corpus.grapheme_encoding)
        pred_idx = predict_word(self.model, transformed, self.corpus.char_vocab, self.device)
        return " ".join(self.corpus.phoneme_vocab.decode(pred_idx))

    def predict_batch(self, words: list) -> list:
        """Lista de palavras → lista de strings de fonemas (loop unitário)."""
        return [self.predict(w) for w in words]

    def predict_batch_native(self, words: list, batch_size: int = 32) -> list:
        """
        Batch inference nativo — processa N palavras por chamada ao modelo.

        Mais eficiente que predict() em loop para throughput:
        1 chamada ao encoder + max_len passos do decoder em paralelo para batch_size items.
        O pipeline garante a mesma saída que predict() chamado individualmente.

        Performance (Exp104d, sweep formal 2026-03-14/15):
          CPU (Xeon 36 cores, MKL):
            batch=32  → 155 w/s  (6.5× vs predict() em loop)
            batch=128 → 190 w/s  (pico CPU; saturação após batch≈64)
          GPU (RTX 3060, CUDA):
            batch=32  → 406 w/s  (11.8× vs predict() em loop)
            batch=128 → 745 w/s
            batch=512 → 1.106 w/s (pico GPU)

        Recomendações:
          - Uso geral / baixo volume: batch_size=32 (boa latência + throughput)
          - Ingestão máxima CPU:      batch_size=128
          - Ingestão máxima GPU:      batch_size=512

        A arquitetura já suporta batch nativamente:
        - Encoder: pack_padded_sequence(enforce_sorted=False)
        - Attention + Decoder: operações batch-first em todos os tensores
        - model.predict(): rastreia finished[b] por item no batch

        Args:
            words:      Lista de palavras grafêmicas.
            batch_size: Items por chamada ao modelo (default: 32).
                        Valores menores reduzem latência máxima; maiores aumentam throughput.

        Returns:
            Lista de strings de fonemas, na mesma ordem de words.
        """
        results = []
        for i in range(0, len(words), batch_size):
            chunk = words[i:i + batch_size]

            # 1. Transformação grafêmica (no-op em 'raw')
            transformed = [
                transform_grapheme_word(w, self.corpus.grapheme_encoding)
                for w in chunk
            ]

            # 2. Codificação char → índices
            encoded = [self.corpus.char_vocab.encode(t) for t in transformed]
            lengths = [len(e) for e in encoded]
            max_char_len = max(lengths)

            # 3. Tensor padded (batch, max_len) — PAD_IDX=0
            padded = torch.zeros(
                len(chunk), max_char_len, dtype=torch.long, device=self.device
            )
            for j, enc in enumerate(encoded):
                padded[j, :len(enc)] = torch.tensor(enc, dtype=torch.long)

            lengths_t = torch.tensor(lengths, dtype=torch.long).to(self.device)

            # 4. Uma chamada ao modelo para todo o chunk
            predictions = self.model.predict(padded, lengths_t, max_len=50)

            # 5. Decodificar índices → fonemas
            for pred_idx in predictions:
                results.append(" ".join(self.corpus.phoneme_vocab.decode(pred_idx)))

        return results

    def predict_stripped(self, word: str) -> str:
        """Como predict(), mas remove separadores silábicos '.' da saída."""
        return " ".join(p for p in self.predict(word).split() if p != ".")

    # -------------------------------------------------------------------------
    # Utilitários
    # -------------------------------------------------------------------------

    @property
    def experiment_name(self) -> str:
        return (self.metadata or {}).get("experiment_name", "desconhecido")

    @property
    def uses_separators(self) -> bool:
        return self.corpus.keep_syllable_separators

    def info(self):
        """Imprime configuração do modelo carregado."""
        meta  = self.metadata or {}
        cfg   = meta.get("config", {})
        m     = cfg.get("model", {})
        d     = cfg.get("data", {})
        best  = meta.get("best_loss") or 0
        loss_t = meta.get("loss_type",
                 cfg.get("training", {}).get("loss", {}).get("type", "cross_entropy"))
        print(f"\nModelo  : {meta.get('experiment_name', '?')}")
        print(f"Run ID  : {meta.get('run_id', '?')}")
        print(f"Epoch   : {meta.get('current_epoch', '?')} | Best loss: {best:.4f}")
        print(f"Params  : {meta.get('total_params', 0):,}")
        print(f"Arch    : emb={m.get('emb_dim','?')} hidden={m.get('hidden_dim','?')} "
              f"layers={m.get('num_layers','?')}")
        print(f"Encoding: {d.get('grapheme_encoding', 'raw')} | "
              f"sep: {d.get('keep_syllable_separators', False)}")
        print(f"Loss    : {loss_t}")
        print(f"Device  : {self.device}")
        print(f"Chars   : {len(self.corpus.char_vocab)} | "
              f"Fonemas: {len(self.corpus.phoneme_vocab)}\n")

    @staticmethod
    def list_models():
        """Lista modelos disponíveis com índice para uso em load(index=N)."""
        models = _get_complete_model_files()
        if not models:
            print("Nenhum modelo encontrado em models/")
            return []
        print(f"\n{'='*70}")
        print("MODELOS DISPONÍVEIS")
        print(f"{'='*70}")
        for i, path in enumerate(models):
            meta   = load_model_metadata(path)
            name   = meta.get("experiment_name", path.stem) if meta else path.stem
            epoch  = meta.get("current_epoch", "?") if meta else "?"
            best   = meta.get("best_loss", 0) if meta else 0
            params = meta.get("total_params", 0) if meta else 0
            sep    = (meta.get("config", {}).get("data", {})
                      .get("keep_syllable_separators", False)) if meta else False
            loss_t = meta.get("loss_type", "ce") if meta else "ce"
            print(f"  [{i:2d}]  {name}")
            print(f"        epoch={epoch} | loss={best:.4f} | {params:,}p | "
                  f"sep={sep} | {loss_t}")
        print(f"{'='*70}")
        print(f"Mais recente: [{len(models)-1}]")
        print("Uso: G2PPredictor.load('best_per') | G2PPredictor.load(index=N)\n")
        return models

    @staticmethod
    def _resolve_alias(alias: str) -> Path:
        """
        Resolve alias semântico ("best_per", "best_wer", "fast") para Path do .pt.

        Lê models/model_registry.json, encontra a entrada do alias, depois localiza
        o arquivo .pt correspondente em models/<experiment>/.
        """
        import json
        registry_path = _PROJECT_ROOT / "models" / "model_registry.json"
        if not registry_path.exists():
            raise FileNotFoundError(
                f"Registry não encontrado: {registry_path}\n"
                "Execute: python src/manage_experiments.py --registry"
            )
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        aliases = registry.get("aliases", {})
        if alias not in aliases:
            available = [k for k, v in aliases.items() if v.get("experiment")]
            raise KeyError(
                f"Alias '{alias}' não encontrado. Disponíveis: {available}\n"
                f"Aliases pendentes (sem checkpoint): "
                f"{[k for k, v in aliases.items() if not v.get('experiment')]}"
            )
        entry = aliases[alias]
        if not entry.get("experiment"):
            raise RuntimeError(
                f"Alias '{alias}' está pendente: {entry.get('description', '')}\n"
                f"Nota: {entry.get('notes', '')}"
            )
        exp_name = entry["experiment"]
        run_id   = entry.get("run_id")
        exp_dir  = _PROJECT_ROOT / "models" / exp_name
        if not exp_dir.exists():
            raise FileNotFoundError(
                f"Pasta do experimento não encontrada: {exp_dir}\n"
                f"O alias '{alias}' aponta para '{exp_name}' que não tem checkpoint local."
            )
        if run_id:
            pt_path = exp_dir / f"{exp_name}__{run_id}.pt"
            if pt_path.exists():
                return pt_path
        # Fallback: qualquer .pt na pasta
        candidates = sorted(exp_dir.glob("*.pt"))
        if not candidates:
            raise FileNotFoundError(
                f"Nenhum .pt encontrado em {exp_dir}\n"
                f"O alias '{alias}' existe no registry mas o checkpoint não está presente."
            )
        return candidates[-1]

    @staticmethod
    def _phoneme_diff(pred: str, ref: str) -> str:
        """
        Diff compacto entre sequências de fonemas (espaço-separados).

        Exemplos:
            _phoneme_diff("t ɪ",   "t ʃ ɪ")   → "faltou 'ʃ'"
            _phoneme_diff("k ɪ",   "k i a x")  → "'ɪ' → 'i a x'"
            _phoneme_diff("ˈ a x", "a x")      → "extra 'ˈ'"
        """
        pred_t  = pred.split()
        ref_t   = ref.split()
        matcher = difflib.SequenceMatcher(None, pred_t, ref_t)
        parts   = []
        for op, i1, i2, j1, j2 in matcher.get_opcodes():
            if op == "equal":
                continue
            elif op == "replace":
                parts.append(
                    f"'{' '.join(pred_t[i1:i2])}' → '{' '.join(ref_t[j1:j2])}'"
                )
            elif op == "insert":
                parts.append(f"faltou '{' '.join(ref_t[j1:j2])}'")
            elif op == "delete":
                parts.append(f"extra '{' '.join(pred_t[i1:i2])}'")
        return " | ".join(parts) if parts else "—"

    @staticmethod
    def _phonological_score(pred: str, ref: str) -> tuple:
        """
        Proximidade fonológica entre predição e referência (0–100%).

        Independente do modelo G2P: usa apenas a tabela _PHONEME_GROUPS,
        que codifica grupos articulatórios para o inventário PT-BR.

        Método: edit distance ponderada (SequenceMatcher) com custo por
        operação dado por _group_dist(). Insertions/deletions custam 1.0.
        Score = 100 × (1 - custo_total / max(|pred|, |ref|)).

        Returns:
            (score: float, label: str)
            score 90-100: "muito próximo" (erros de detalhe: vozeamento, altura)
            score 70-89 : "próximo"       (alguns fonemas trocados)
            score 50-69 : "parcial"       (estrutura parcialmente correta)
            < 50        : "distante"      (erros estruturais graves)
        """
        if pred == ref:
            return 100.0, "exato"
        pred_t = pred.split()
        ref_t  = ref.split()
        if not pred_t or not ref_t:
            return 0.0, "vazio"

        n_max   = max(len(pred_t), len(ref_t))
        matcher = difflib.SequenceMatcher(None, pred_t, ref_t)
        cost    = 0.0

        for op, i1, i2, j1, j2 in matcher.get_opcodes():
            if op == "equal":
                pass
            elif op == "replace":
                p_span = pred_t[i1:i2]
                r_span = ref_t[j1:j2]
                for p, r in zip(p_span, r_span):
                    cost += _group_dist(p, r)
                # Extra tokens na span mais longa → custo de inserção/deleção
                cost += abs(len(p_span) - len(r_span))
            elif op == "insert":
                cost += j2 - j1
            elif op == "delete":
                cost += i2 - i1

        score = max(0.0, 100.0 * (1.0 - cost / n_max))
        if score >= 90:
            label = "muito próximo"
        elif score >= 70:
            label = "próximo"
        elif score >= 50:
            label = "parcial"
        else:
            label = "distante"
        return score, label

    def _char_coverage(self, word: str) -> tuple:
        """
        Verifica cobertura de caracteres do word no char_vocab do modelo.

        Chars OOV (não vistos no treino) causam mapeamento para <UNK>,
        prejudicando diretamente a predição.

        Returns:
            (coverage: float 0–100, oov_chars: set[str])
        """
        known     = {c for c in self.corpus.char_vocab.c2i if not c.startswith("<")}
        word_low  = word.lower()
        unique    = set(word_low)
        oov       = unique - known
        coverage  = 100.0 * (1.0 - len(oov) / max(len(unique), 1))
        return coverage, oov

    def find_and_compare_similar(self, word: str, n_similar: int = 5) -> None:
        """
        Encontra palavras similares (ortograficamente) no dicionário e compara
        suas pronúncias com a palavra consultada.

        Útil para entender como o modelo trata variações fonológicas e
        padrões de pronúncia em famílias de palavras.

        Args:
            word: Palavra a buscar (ex: "laeta")
            n_similar: Quantas palavras similares retornar (default: 5)
        """
        word_lower = word.lower()

        # Buscar palavras similares no corpus
        candidates = difflib.get_close_matches(
            word_lower,
            [w.lower() for w in self.corpus.words_raw],
            n=n_similar + 1,  # +1 para possível match exato
            cutoff=0.6
        )

        # Remover a palavra consultada se estiver na lista
        candidates = [c for c in candidates if c != word_lower][:n_similar]

        if not candidates:
            print(f"\n[Nenhuma palavra similar encontrada para '{word}']")
            return

        # Cabeçalho
        print(f"\n{'='*75}")
        print(f"Comparação de Pronúncias — Palavras similares a '{word}'")
        print(f"Modelo: {self.experiment_name}")
        print(f"{'='*75}")
        print(f"  {'Palavra':<20} {'Pronúncia':<35} Aproximação")
        print(f"  {'-'*72}")

        # Predição da palavra consultada
        pred_word = self.predict(word_lower)
        print(f"  {word_lower:<20} {pred_word:<35} [CONSULTA]")
        print(f"  {'-'*72}")

        # Comparar com similares
        for candidate in candidates:
            pred_candidate = self.predict(candidate)
            score, label = self._phonological_score(pred_candidate, pred_word)

            # Formatar para visualização
            score_str = f"{score:.0f}% ({label})"
            print(f"  {candidate:<20} {pred_candidate:<35} {score_str}")

        print(f"{'='*75}\n")

    # =========================================================================
    # PARTE 2 — AVALIAÇÃO COM REFERÊNCIA (opcional)
    #
    # Útil quando você tem um TSV com fonemas esperados e quer comparar
    # as predições do modelo com a referência (ex: conjunto de neologismos).
    #
    # O TSV pode ter N colunas — apenas as duas primeiras são usadas:
    #   col 0: palavra grafêmica
    #   col 1: fonemas de referência (espaço-separados)
    # Linhas começando com '#' são ignoradas (comentários/cabeçalho).
    # =========================================================================

    def evaluate_tsv(self, tsv_path: "str | Path",
                     cache_tag: str = "",
                     cache_dir: "str | Path | None" = None) -> dict:
        """
        Carrega um TSV com palavras + referências, infere e calcula métricas.

        Fluxo:
          1. G2PCorpus carrega o TSV — garante a mesma transformação grafêmica
             usada no treino (transform_grapheme_word).
          2. split(test_ratio=1.0) coloca todas as palavras em 'test'.
             Se cache_dir for dado, salva cache com nome identificável via cache_tag.
          3. self.predict() infere cada palavra usando os vocabs do modelo.
          4. WER/PER calculados com as mesmas funções de inference.py.

        Args:
            tsv_path:  Caminho do TSV (ex: "docs/data/neologisms_test.tsv").
            cache_tag: Prefixo do cache para identificação fácil.
                       Ex: "neologismos" → "data/neologismos_test_raw_nosep_0-0-100_s42.txt"
            cache_dir: Onde salvar o cache gerado. None = sem cache.

        Returns:
            dict com "per", "wer", "accuracy", "predictions", "words_raw".
        """
        tsv_path = Path(tsv_path)
        if not tsv_path.exists():
            raise FileNotFoundError(f"TSV não encontrado: {tsv_path}")

        # PASSO 1 — Carregar TSV pelo G2PCorpus
        #   Aceita N colunas; usa apenas as 2 primeiras (palavra, fonemas).
        #   Aplica transform_grapheme_word com as mesmas configs do modelo,
        #   garantindo consistência com o pipeline de treino.
        ext_corpus = G2PCorpus(
            tsv_path,
            grapheme_encoding=self.corpus.grapheme_encoding,
            keep_syllable_separators=self.corpus.keep_syllable_separators,
        )

        # PASSO 2 — Split 100% test; gera cache com nome identificável
        split = ext_corpus.split(
            test_ratio=1.0,
            cache_dir=cache_dir,
            cache_tag=cache_tag,
        )

        # Palavras brutas (para exibição e para predict()) e referências
        words_raw = [ext_corpus.words_raw[i] for i in split.test_indices]
        _, refs   = split.test_pairs()   # refs = lista de strings espaçadas

        # PASSO 3 — Inferência
        #   self.predict(palavra_bruta) aplica transform_grapheme_word internamente
        #   e usa os vocabs do modelo (pt-br.tsv), não do TSV externo.
        predictions = [self.predict(w) for w in words_raw]

        # PASSO 4 — Exibir comparação palavra a palavra
        print(f"\n{'='*65}")
        print(f"Avaliação : {tsv_path.name} | {len(words_raw)} palavras")
        print(f"Modelo    : {self.experiment_name}")
        print(f"{'='*65}")
        print(f"  {'Palavra':<22} {'Predição':<30} Referência")
        print(f"  {'-'*62}")
        for word, pred, ref in zip(words_raw, predictions, refs):
            ok = "✓" if pred == ref else " "
            print(f"{ok} {word:<22} {pred:<30} {ref}")

        # PASSO 5 — Métricas (mesmas funções de inference.py)
        pred_lists = [p.split() for p in predictions]
        ref_lists  = [r.split() for r in refs]
        per      = calculate_per(pred_lists, ref_lists)
        wer      = calculate_wer(pred_lists, ref_lists)
        accuracy = calculate_accuracy(pred_lists, ref_lists)
        correct  = sum(1 for p, r in zip(predictions, refs) if p == r)

        print(f"\n  {'─'*62}")
        print(f"  PER: {per:.2f}%  |  WER: {wer:.2f}%  |  Accuracy: {accuracy:.2f}%")
        print(f"  Corretas: {correct}/{len(words_raw)}")
        print(f"{'='*65}\n")

        return {
            "per": per, "wer": wer, "accuracy": accuracy,
            "predictions": predictions, "words_raw": words_raw,
        }

    # =========================================================================
    # PARTE 3 — AVALIAÇÃO DE NEOLOGISMOS (análise rica por categoria)
    #
    # Usa o TSV estendido (5 colunas): word, phonemes, category, difficulty, notes.
    # Para cada falha, exibe diff fonêmico e a nota linguística do TSV.
    # As referências do TSV não têm separadores silábicos, então predict_stripped()
    # é usado automaticamente (independente do modelo).
    # =========================================================================

    def evaluate_neologisms(self, tsv_path: "str | Path" = None) -> dict:
        """
        Avalia banco de neologismos com análise fonológica de falhas.

        Lê TSV estendido (5 colunas: word, phonemes, category, difficulty, notes),
        prediz cada palavra e exibe análise detalhada das erradas — diff fonêmico
        e nota linguística do próprio TSV.

        As referências não têm separadores silábicos ('.'): predict_stripped() é
        usado automaticamente independente do modelo carregado.

        Args:
            tsv_path: Caminho do TSV. Padrão: docs/data/neologisms_test.tsv

        Returns:
            dict com 'per', 'wer', 'accuracy', 'results', 'by_category', 'by_difficulty'
        """
        if tsv_path is None:
            tsv_path = _PROJECT_ROOT / "docs" / "data" / "neologisms_test.tsv"
        tsv_path = Path(tsv_path)
        if not tsv_path.exists():
            raise FileNotFoundError(f"TSV não encontrado: {tsv_path}")

        # Ler todas as colunas do TSV diretamente
        rows = []
        with open(tsv_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split("\t")
                if len(parts) < 2:
                    continue
                rows.append({
                    "word":       parts[0],
                    "ref":        parts[1],
                    "category":   parts[2] if len(parts) > 2 else "",
                    "difficulty": parts[3] if len(parts) > 3 else "",
                    "note":       parts[4] if len(parts) > 4 else "",
                })

        if not rows:
            print("Nenhuma palavra encontrada no TSV.")
            return {}

        # Inferir — sempre sem separadores (referências do TSV não têm '.')
        # Lowercase antes de prever: modelo treinado em minúsculas.
        # A forma original é preservada em r["word"] para exibição.
        for r in rows:
            r["pred"]    = self.predict_stripped(r["word"].lower())
            r["correct"] = (r["pred"] == r["ref"])
            r["em_dict"] = "[EM DICT]" in r["note"]

        # Métricas globais
        pred_lists = [r["pred"].split() for r in rows]
        ref_lists  = [r["ref"].split()  for r in rows]
        per        = calculate_per(pred_lists, ref_lists)
        wer        = calculate_wer(pred_lists, ref_lists)
        accuracy   = calculate_accuracy(pred_lists, ref_lists)
        n_correct  = sum(1 for r in rows if r["correct"])

        # Agrupar por categoria — ordem canônica + novas categorias dinamicamente
        cat_order = [
            "anglicismo", "verbo_emprestimo", "neologismo_nativo",
            "nome_proprio", "palavra_inventada",
            "generalizacao_pt", "consoante_dupla", "anglicismo_invocab",
            "char_oov", "real_oov", "controle",
        ]
        categories = {}
        for cat in cat_order:
            categories[cat] = [r for r in rows if r["category"] == cat]
        for r in rows:  # categorias não previstas pela ordem canônica
            if r["category"] not in categories:
                categories.setdefault(r["category"], []).append(r)

        # Enriquecer: calcular score e cobertura de chars para cada linha
        for r in rows:
            score, slabel        = self._phonological_score(r["pred"], r["ref"])
            cov, oov             = self._char_coverage(r["word"])
            r["phon_score"]  = score
            r["phon_label"]  = slabel
            r["char_cov"]    = cov
            r["char_oov"]    = oov

        # Exibir
        SEP = "═" * 72
        cat_labels = {
            "anglicismo":          "Anglicismos",
            "verbo_emprestimo":    "Verbos Emprestados",
            "neologismo_nativo":   "Neologismos Nativos",
            "nome_proprio":        "Nomes Próprios",
            "palavra_inventada":   "Palavras Inventadas",
            "generalizacao_pt":    "Generalização PT-BR",
            "consoante_dupla":     "Consoantes Duplas",
            "anglicismo_invocab":  "Anglicismos (chars no vocab)",
            "char_oov":            "Chars OOV (k / w / y)",
            "real_oov":            "Palavras PT-BR (prováveis OOV)",
            "controle":            "Controles (em treino)",
        }
        print(f"\n{SEP}")
        print(f"  Avaliação: {tsv_path.name}")
        print(f"  Modelo  : {self.experiment_name}")
        print(f"  Total   : {len(rows)} palavras | {n_correct}/{len(rows)} corretas "
              f"({100 * n_correct / len(rows):.0f}%)")
        print(SEP)

        by_cat = {}
        for cat, cat_rows in categories.items():
            if not cat_rows:
                continue
            n_ok      = sum(1 for r in cat_rows if r["correct"])
            label     = cat_labels.get(cat, cat.replace("_", " ").title())
            avg_score = sum(r["phon_score"] for r in cat_rows) / len(cat_rows)
            print(f"\n  ── {label.upper()} ({n_ok}/{len(cat_rows)} corretas | "
                  f"aprox. fonol. média: {avg_score:.0f}%)")

            for r in cat_rows:
                ctrl = " [controle]" if r["em_dict"] else ""
                ok   = "✓" if r["correct"] else "✗"
                tag  = f"[{r['difficulty']}]" if r["difficulty"] else ""
                # Alerta OOV nos chars (mesmo para corretas — informativo)
                oov_info = f"  ⚠ OOV: {', '.join(sorted(r['char_oov']))}" \
                           if r["char_oov"] else ""
                if r["correct"]:
                    print(f"  {ok}  {r['word']:<22} {tag:<8}  {r['pred']}{ctrl}{oov_info}")
                else:
                    score_str = f"{r['phon_score']:.0f}% {r['phon_label']}"
                    print(f"  {ok}  {r['word']:<22} {tag:<8}  "
                          f"[fonol: {score_str}]{ctrl}{oov_info}")
                    print(f"       Predito : {r['pred']}")
                    print(f"       Esperado: {r['ref']}")
                    diff_str = self._phoneme_diff(r["pred"], r["ref"])
                    print(f"       Diff    : {diff_str}")
                    if r["note"]:
                        note_clean = (r["note"]
                                      .replace("[EM DICT] controle: ", "")
                                      .replace("[EM DICT] ", "")
                                      .replace("[CHAR OOV: ", "⚠ [OOV: "))
                        print(f"       Nota    : {note_clean}")

            by_cat[cat] = {
                "total":    len(cat_rows),
                "correct":  n_ok,
                "accuracy": 100 * n_ok / len(cat_rows) if cat_rows else 0,
            }

        # Sumário por dificuldade
        diffs_seen: dict = {}
        for r in rows:
            d = r["difficulty"] or "desconhecida"
            diffs_seen.setdefault(d, {"total": 0, "correct": 0})
            diffs_seen[d]["total"] += 1
            if r["correct"]:
                diffs_seen[d]["correct"] += 1

        print(f"\n  {'─' * 68}")
        print("  Por dificuldade:")
        for d in ["easy", "medium", "hard"]:
            if d not in diffs_seen:
                continue
            stats = diffs_seen[d]
            pct   = 100 * stats["correct"] / stats["total"]
            bar   = "█" * int(pct / 10) + "░" * (10 - int(pct / 10))
            print(f"    {d:<8}: {stats['correct']:2}/{stats['total']:2}  {bar}  {pct:.0f}%")

        print(f"\n  PER: {per:.2f}%  |  WER: {wer:.2f}%  |  Accuracy: {accuracy:.2f}%")
        print(f"  Corretas: {n_correct}/{len(rows)}")
        print(f"{SEP}\n")

        return {
            "per":           per,
            "wer":           wer,
            "accuracy":      accuracy,
            "results":       rows,
            "by_category":   by_cat,
            "by_difficulty": diffs_seen,
        }


# =============================================================================
# CLI — linha de comando
# =============================================================================

def _cli():
    parser = argparse.ArgumentParser(
        description="inference_light — tutorial G2P: palavra → fonemas IPA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos básicos (inferência):
  python src/inference_light.py --word computador
  python src/inference_light.py --index 11 --words "selfie,drone,blog"
  python src/inference_light.py --interactive
  python src/inference_light.py --list

Avaliação de neologismos (análise rica de falhas + score fonológico):
  python src/inference_light.py --neologisms
  python src/inference_light.py --index 18 --neologisms
  python src/inference_light.py --index 18 --neologisms docs/data/generalization_test.tsv

Com avaliação genérica (TSV com referências):
  python src/inference_light.py --tsv docs/data/neologisms_test.tsv
  python src/inference_light.py --index 11 --tsv docs/data/neologisms_test.tsv \\
      --cache-tag neologismos --cache-dir data

Avaliação completa (WER/PER no test set padrão):
  python src/inference.py --index 11
        """,
    )
    parser.add_argument("--alias", type=str, default=None,
                        help="Alias semantico: best_per, best_wer, fast (ver model_registry.json).")
    parser.add_argument("--index", type=int, default=None,
                        help="Índice do modelo (ver --list). Padrão: mais recente.")
    parser.add_argument("--model", type=str, default=None,
                        help="Nome do arquivo de modelo (sem .pt).")
    parser.add_argument("--word", type=str, default=None,
                        help="Palavra única.")
    parser.add_argument("--words", type=str, default=None,
                        help="Lista de palavras separadas por vírgula.")
    parser.add_argument("--file", type=str, default=None,
                        help="Arquivo .txt com uma palavra por linha.")
    parser.add_argument("--tsv", type=str, default=None,
                        help="TSV com palavra + fonemas de referência (avaliação opcional).")
    parser.add_argument("--cache-tag", type=str, default="",
                        help="Prefixo do cache ao usar --tsv (ex: 'neologismos').")
    parser.add_argument("--cache-dir", type=str, default=None,
                        help="Diretório para salvar cache do --tsv (ex: 'data').")
    parser.add_argument("--interactive", action="store_true",
                        help="Modo interativo: digita palavras e vê fonemas.")
    parser.add_argument("--list", action="store_true",
                        help="Lista modelos disponíveis.")
    parser.add_argument("--info", action="store_true",
                        help="Mostra configuração do modelo.")
    parser.add_argument("--strip-sep", action="store_true",
                        help="Remove '.' (separadores silábicos) da saída.")
    parser.add_argument("--neologisms", nargs="?", const="__DEFAULT__",
                        metavar="TSV_PATH",
                        help="Avalia TSV de neologismos com análise de falhas, "
                             "score fonológico e detecção de chars OOV. "
                             "Sem argumento: usa docs/data/neologisms_test.tsv. "
                             "Com argumento: --neologisms docs/data/generalization_test.tsv")
    parser.add_argument("--similar", action="store_true",
                        help="Encontra palavras similares (ortograficamente) no dicionário "
                             "e compara suas pronúncias. Requer --word. "
                             "Ex: python src/inference_light.py --word laeta --similar")
    parser.add_argument("--similar-count", type=int, default=5, metavar="N",
                        help="Número de palavras similares a retornar (default: 5). "
                             "Usado com --similar.")
    parser.add_argument("--quantize", action="store_true",
                        help="Ativa INT8 dynamic quantization no CPU. "
                             "ATENCAO: em Xeon+Windows causa regressao de 38%%. "
                             "Util apenas para teste em ARM — ver docs/evaluations/answered/016.")
    parser.add_argument("--threads", type=int, default=None, metavar="N",
                        help="Threads intra-op do PyTorch para CPU (default: sistema/MKL). "
                             "ATENCAO: em Xeon, threads=1 causou regressao de 61%% — MKL "
                             "multi-thread ja e otimo para matrizes LSTM hidden=384. "
                             "Util apenas para teste em hardware sem MKL otimizado.")
    parser.add_argument("--batch-size", type=int, default=32, metavar="N",
                        help="Batch size para --words e --file (default: 32). "
                             "Aumenta throughput: CPU pico batch=128 (~190 w/s); "
                             "GPU pico batch=512 (~1.106 w/s). "
                             "Para palavra única (--word), sem efeito.")

    args = parser.parse_args()

    has_action = any([args.word, args.words, args.file, args.tsv,
                      args.neologisms is not None,
                      args.interactive, args.info, args.similar])
    if args.list or not has_action:
        G2PPredictor.list_models()
        return

    model_path = MODELS_DIR / f"{args.model}.pt" if args.model else None
    predictor  = G2PPredictor.load(
        alias=args.alias, model_path=model_path, index=args.index,
        quantize=args.quantize,
        num_threads=args.threads,
    )

    if args.info:
        predictor.info()
        return

    # Avaliação de neologismos/generalização com análise detalhada de falhas
    if args.neologisms is not None:
        tsv = None if args.neologisms == "__DEFAULT__" else args.neologisms
        predictor.evaluate_neologisms(tsv_path=tsv)
        return

    # Avaliação com TSV genérico (opcional — só se --tsv for passado)
    if args.tsv:
        predictor.evaluate_tsv(
            tsv_path=args.tsv,
            cache_tag=args.cache_tag,
            cache_dir=args.cache_dir,
        )
        return

    # Função auxiliar de predição (com ou sem strip)
    def _predict(word: str) -> str:
        return predictor.predict_stripped(word.strip()) if args.strip_sep \
               else predictor.predict(word.strip())

    if args.word:
        if args.similar:
            # Comparar com palavras similares
            predictor.find_and_compare_similar(args.word, n_similar=args.similar_count)
        else:
            # Apenas predizer
            print(f"{args.word}\t{_predict(args.word)}")
        return

    if args.words:
        word_list = [w.strip() for w in args.words.split(",") if w.strip()]
        phoneme_list = predictor.predict_batch_native(word_list, batch_size=args.batch_size)
        if args.strip_sep:
            phoneme_list = [" ".join(p for p in ph.split() if p != ".") for ph in phoneme_list]
        for w, ph in zip(word_list, phoneme_list):
            print(f"{w}\t{ph}")
        return

    if args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"Arquivo não encontrado: {file_path}")
            sys.exit(1)
        word_list = [ln.strip() for ln in file_path.read_text(encoding="utf-8").splitlines()
                     if ln.strip()]
        phoneme_list = predictor.predict_batch_native(word_list, batch_size=args.batch_size)
        if args.strip_sep:
            phoneme_list = [" ".join(p for p in ph.split() if p != ".") for ph in phoneme_list]
        for w, ph in zip(word_list, phoneme_list):
            print(f"{w}\t{ph}")
        return

    if args.interactive:
        sep_info = " (com separadores silábicos)" if predictor.uses_separators else ""
        print(f"\nModo interativo — {predictor.experiment_name}{sep_info}")
        print("Digite uma palavra e pressione Enter. 'sair' ou Ctrl+C para encerrar.\n")
        try:
            while True:
                word = input("  Palavra: ").strip()
                if not word or word.lower() in ("sair", "exit", "quit"):
                    break
                print(f"  Fonemas: {_predict(word)}\n")
        except (KeyboardInterrupt, EOFError):
            print("\nEncerrando.")


if __name__ == "__main__":
    _cli()
