import hashlib
import logging
import math
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import stats as sp_stats
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from sklearn.model_selection import train_test_split

from phoneme_embeddings import get_embedding_layer

logger = logging.getLogger("g2p")

GRAPHEME_MARKER_MAP = {
    "\u0301": "'",   # acute
    "\u0300": "`",   # grave
    "\u0302": "^",   # circumflex
    "\u0303": "~",   # tilde
    "\u0327": "'",   # cedilla
    "\u0308": ":",   # diaeresis
}


class GraphemeConfig:
    """Configuração de transformação grafêmica com suporte a filtros.

    Formatos suportados:
    1. "raw" — sem transformação
    2. "decomposed" — decomposição NFD com marcas de diacríticos
    3. {"type": "raw", "filters": ["-", "."]} — raw + remove caracteres
    4. {"type": "decomposed", "filters": ["-"]} — decomposed + remove caracteres
    """

    def __init__(self, encoding: str | dict = "raw"):
        if isinstance(encoding, str):
            self.type = encoding
            self.filters = []
        elif isinstance(encoding, dict):
            self.type = encoding.get("type", "raw")
            self.filters = encoding.get("filters", [])
        else:
            raise ValueError(f"grapheme_encoding deve ser str ou dict, não {type(encoding)}")

        if self.type not in ["raw", "decomposed"]:
            raise ValueError(f"type deve ser 'raw' ou 'decomposed', não {self.type!r}")

    def __repr__(self):
        if self.filters:
            return f"GraphemeConfig(type={self.type!r}, filters={self.filters})"
        return f"GraphemeConfig(type={self.type!r})"

    def transform(self, word: str) -> str:
        """Aplica transformação grafêmica + filtros."""
        # Step 1: Transformação base (raw ou decomposed)
        if self.type == "raw":
            result = word
        else:  # decomposed
            decomposed = unicodedata.normalize("NFD", word)
            out = []
            pending_marks = []
            for ch in decomposed:
                if unicodedata.combining(ch):
                    marker = GRAPHEME_MARKER_MAP.get(ch, "")
                    if marker:
                        pending_marks.append(marker)
                    continue
                if pending_marks:
                    out.extend(pending_marks)
                    pending_marks.clear()
                out.append(ch)
            result = "".join(out)

        # Step 2: Aplicar filtros (remover caracteres)
        for char_to_remove in self.filters:
            result = result.replace(char_to_remove, "")

        return result


def transform_grapheme_word(word: str, encoding: str | dict = "raw") -> str:
    """Transforma palavra grafêmica conforme estratégia de codificação.

    Compatível com:
    - str: "raw" ou "decomposed"
    - dict: {"type": "raw", "filters": [...]}
    """
    config = GraphemeConfig(encoding)
    return config.transform(word)


# ---------------------------------------------------------------------------
# Vocabulários
# ---------------------------------------------------------------------------
# PAD = 0  → padding para alinhar batches. Embedding fixo em zero (padding_idx),
#            ignorado pela loss (ignore_index). Não é aprendido.
# UNK = 1  → fallback para tokens fora do vocabulário. Embedding aprendido.
# EOS = 2  → (só no PhonemeVocab) marca fim da sequência de saída.
#            É um embedding APRENDIDO: o decoder precisa aprender a predizê-lo
#            na posição correta. Na inferência, geração para quando EOS é emitido.
#            Adicionado automaticamente ao target pelo G2PDataset.__getitem__.
# ---------------------------------------------------------------------------

class CharVocab:
    """Vocabulário de caracteres (entrada). PAD=0, UNK=1."""
    def __init__(self):
        self.c2i = {'<PAD>': 0, '<UNK>': 1}
        self.i2c = {0: '<PAD>', 1: '<UNK>'}

    def add(self, chars):
        for c in chars:
            if c not in self.c2i:
                idx = len(self.c2i)
                self.c2i[c] = idx
                self.i2c[idx] = c

    def encode(self, text):
        return [self.c2i.get(c, self.c2i['<UNK>']) for c in text]

    def decode(self, indices):
        return ''.join(self.i2c.get(i, '?') for i in indices)

    def __len__(self):
        return len(self.c2i)


class PhonemeVocab:
    """Vocabulário de fonemas (saída). PAD=0, UNK=1, EOS=2."""
    PAD_IDX = 0
    UNK_IDX = 1
    EOS_IDX = 2

    def __init__(self):
        self.p2i = {'<PAD>': 0, '<UNK>': 1, '<EOS>': 2}
        self.i2p = {0: '<PAD>', 1: '<UNK>', 2: '<EOS>'}

    def add(self, phonemes):
        for p in phonemes:
            if p not in self.p2i:
                idx = len(self.p2i)
                self.p2i[p] = idx
                self.i2p[idx] = p

    def encode(self, phoneme_str):
        phonemes = phoneme_str.split() if isinstance(phoneme_str, str) else phoneme_str
        return [self.p2i.get(p, self.p2i['<UNK>']) for p in phonemes]

    def decode(self, indices):
        """Decodifica índices parando no primeiro EOS (ou fim). Ignora PAD."""
        phonemes = []
        for i in indices:
            if i == self.EOS_IDX:
                break
            if i == self.PAD_IDX:
                continue
            phonemes.append(self.i2p.get(i, '<UNK>'))
        return phonemes

    def __len__(self):
        return len(self.p2i)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class G2PDataset(Dataset):
    """
    Dataset G2P. Retorna (chars_tensor, phonemes_tensor).
    
    O EOS é adicionado automaticamente ao final da sequência de fonemas.
    Isso é interno ao treinamento — os arquivos de dados NÃO contêm EOS.
    """
    def __init__(self, words, phonemes, char_vocab, phoneme_vocab):
        self.words = words
        self.phonemes = phonemes
        self.char_vocab = char_vocab
        self.phoneme_vocab = phoneme_vocab

    def __len__(self):
        return len(self.words)

    def __getitem__(self, idx):
        word = self.words[idx]
        phoneme_seq = self.phonemes[idx]

        chars = torch.LongTensor(self.char_vocab.encode(word))
        phon_indices = self.phoneme_vocab.encode(phoneme_seq)
        # EOS é adicionado aqui — é um token interno do modelo, não do dataset
        phon_indices.append(PhonemeVocab.EOS_IDX)
        phonemes = torch.LongTensor(phon_indices)

        return chars, phonemes


def collate_fn(batch):
    """Agrupa batch com padding. Também retorna comprimentos reais dos inputs."""
    chars, phonemes = zip(*batch)
    char_lengths = torch.LongTensor([len(c) for c in chars])
    chars = pad_sequence(list(chars), batch_first=True, padding_value=0)
    phonemes = pad_sequence(list(phonemes), batch_first=True, padding_value=0)
    return chars, phonemes, char_lengths


def get_dataloaders(words_train, phonemes_train, words_test, phonemes_test,
                    char_vocab, phoneme_vocab, batch_size=32):
    """Legacy helper — prefira G2PCorpus.split().get_dataloaders()."""
    train_ds = G2PDataset(words_train, phonemes_train, char_vocab, phoneme_vocab)
    test_ds = G2PDataset(words_test, phonemes_test, char_vocab, phoneme_vocab)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                         collate_fn=collate_fn)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                        collate_fn=collate_fn)

    return train_dl, test_dl


# ---------------------------------------------------------------------------
# Corpus: fonte primária e gestão de splits
# ---------------------------------------------------------------------------


def _extract_features(word: str, phoneme_str: str) -> dict:
    """
    Extrai features linguísticas de um par (palavra, fonemas).
    Usado para criar strata de estratificação.

    Features extraídas:
    - stress_type:  oxítona / paroxítona / proparoxítona / other
    - syllable_bin: agrupamento por nº de sílabas (1, 2, 3, 4, 5, 6+)
    - length_bin:   agrupamento por comprimento da palavra (1-4, 5-7, 8-10, 11+)
    - ratio_bin:    razão fonemas/caracteres (<0.8, 0.8-1.0, 1.0-1.2, 1.2+)
    """
    tokens = phoneme_str.split()

    # --- nº de sílabas (separador '.' no TSV original) ---
    n_syl = tokens.count(".") + 1
    if n_syl <= 2:
        syl_bin = "syl1-2"
    elif n_syl <= 4:
        syl_bin = "syl3-4"
    else:
        syl_bin = "syl5+"

    # --- posição da tônica ---
    syl_idx = 0
    stress_syl = -1
    for t in tokens:
        if t == ".":
            syl_idx += 1
        elif t == "\u02c8":  # ˈ
            stress_syl = syl_idx

    if stress_syl < 0:
        stress_type = "none"
    else:
        from_end = n_syl - 1 - stress_syl
        if from_end == 0:
            stress_type = "oxi"
        elif from_end == 1:
            stress_type = "parox"
        elif from_end == 2:
            stress_type = "proparox"
        else:
            stress_type = "other"

    # --- comprimento da palavra ---
    wl = len(word)
    if wl <= 4:
        len_bin = "w1-4"
    elif wl <= 7:
        len_bin = "w5-7"
    elif wl <= 10:
        len_bin = "w8-10"
    else:
        len_bin = "w11+"

    # --- ratio fonema / caractere ---
    clean_phon = [t for t in tokens if t not in (".", "\u02c8")]
    ratio = len(clean_phon) / max(wl, 1)
    if ratio < 0.8:
        ratio_bin = "r<0.8"
    elif ratio < 1.0:
        ratio_bin = "r0.8"
    elif ratio < 1.2:
        ratio_bin = "r1.0"
    else:
        ratio_bin = "r1.2+"

    return {
        "stress_type": stress_type,
        "syllable_bin": syl_bin,
        "length_bin": len_bin,
        "ratio_bin": ratio_bin,
    }


def _build_stratum_labels(words: list[str], phonemes_raw: list[str]) -> list[str]:
    """
    Constrói etiqueta composta de estrato para cada par (palavra, fonema_raw).
    phonemes_raw = fonemas do TSV original (com '.' e 'ˈ').

    A etiqueta combina stress × sílabas × comprimento, produzindo ~48 estratos.
    ratio_bin NÃO entra na key para evitar fragmentação excessiva que quebraria
    o stratified split com poucos exemplos por estrato.
    """
    labels = []
    for w, p in zip(words, phonemes_raw):
        feat = _extract_features(w, p)
        # Combinação hierárquica: stress > syllable > length
        label = f"{feat['stress_type']}|{feat['syllable_bin']}|{feat['length_bin']}"
        labels.append(label)
    return labels


class G2PCorpus:
    """
    Fonte primária do dataset G2P.  Lê dicts/*.tsv e gerencia tudo:
    vocabulários, splits determinísticos com seed, cache em data/ para
    inspeção humana, e metadados (checksum + params) para reprodutibilidade.

    O split é ESTRATIFICADO: distribui uniformemente stress position,
    contagem de sílabas e comprimento de palavra entre treino e teste.

    Uso:
        corpus = G2PCorpus("dicts/pt-br.tsv")
        split  = corpus.split(test_ratio=0.2, seed=42, cache_dir="data")
        train_dl, test_dl = split.get_dataloaders(batch_size=32)
        split.log_quality()  # telemetria

    O inference.py recria o mesmo split lendo os metadados do modelo:
        corpus = G2PCorpus("dicts/pt-br.tsv")
        corpus.verify(metadata["dict_checksum"])
        split  = corpus.split(
            test_ratio=metadata["test_ratio"],
            seed=metadata["seed"],
        )
        test_words, test_phonemes = split.test_pairs()
    """

    def __init__(self, dict_path: str | Path, grapheme_encoding: str | dict = "raw",
                 keep_syllable_separators: bool = False):
        self.dict_path = Path(dict_path).resolve()
        if not self.dict_path.exists():
            raise FileNotFoundError(f"Dicionário não encontrado: {self.dict_path}")

        # Parse e valida grapheme_encoding (str ou dict)
        self.grapheme_config = GraphemeConfig(grapheme_encoding)
        self.grapheme_encoding = grapheme_encoding  # salva original para logs/metadata
        self.keep_syllable_separators = keep_syllable_separators

        self.words_raw, self.words, self.phonemes, self._phonemes_raw = self._load_tsv()
        self.checksum = self._compute_checksum()

        # Vocabulários construídos a partir de TODO o corpus (não só do treino)
        # para garantir que nenhum fonema do teste seja OOV.
        self.char_vocab = CharVocab()
        self.phoneme_vocab = PhonemeVocab()
        for w in self.words:
            self.char_vocab.add(w)
        for p in self.phonemes:
            self.phoneme_vocab.add(p.split())

        # Feature labels para estratificação
        self._strata = _build_stratum_labels(self.words_raw, self._phonemes_raw)

        logger.info(
            "Corpus carregado: %d palavras | checksum=%s | chars=%d | phonemes=%d | estratos=%d | grapheme_config=%s | syllable_separators=%s",
            len(self.words), self.checksum,
            len(self.char_vocab), len(self.phoneme_vocab),
            len(set(self._strata)), str(self.grapheme_config),
            "keep" if self.keep_syllable_separators else "remove",
        )

    # ---- carregamento ----

    def _load_tsv(self):
        """Carrega dicionário do arquivo TSV.

        Retorna (words_raw, words_transformed, phonemes_clean, phonemes_raw).
        """
        words_raw, words_transformed, phonemes_clean, phonemes_raw = [], [], [], []
        with open(self.dict_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split("\t")
                if len(parts) < 2:
                    continue
                word = parts[0].strip()
                transformed_word = self.grapheme_config.transform(word)
                phon_raw = parts[1].strip()
                # Clean: controla se separadores de sílaba são mantidos ou removidos
                if self.keep_syllable_separators:
                    phon = phon_raw
                else:
                    phon = phon_raw.replace(". ", " ").replace(".", "").strip()
                phon = " ".join(phon.split())
                # NFC: converte combining tilde (a + U+0303) → precomposta (ã)
                # Sem isso, tokens como 'ã' ficam com 2 codepoints e quebram
                # comparações na inferência.
                phon = unicodedata.normalize("NFC", phon)
                if word and phon:
                    words_raw.append(word)
                    words_transformed.append(transformed_word)
                    phonemes_clean.append(phon)
                    phonemes_raw.append(phon_raw)
        return words_raw, words_transformed, phonemes_clean, phonemes_raw

    def _compute_checksum(self) -> str:
        """SHA-256 truncado (16 chars) do conteúdo bruto do TSV."""
        h = hashlib.sha256(self.dict_path.read_bytes())
        return h.hexdigest()[:16]

    def verify(self, expected_checksum: str):
        """Valida que o dicionário atual é o mesmo usado no treinamento."""
        if self.checksum != expected_checksum:
            raise ValueError(
                f"Checksum do dicionário mudou!\n"
                f"  Esperado : {expected_checksum}\n"
                f"  Atual    : {self.checksum}\n"
                f"O dicionário foi modificado desde o treinamento."
            )

    # ---- split ----

    def split(self, test_ratio: float = 0.2, val_ratio: float = 0.1,
              seed: int = 42,
              cache_dir: str | Path | None = None,
              cache_tag: str = "") -> "DataSplit":
        """
        Cria split ESTRATIFICADO 3-way: train / val / test.

        Estratifica por (tonicidade × sílabas × comprimento), garantindo
        que cada família linguística esteja representada proporcionalmente
        nos três subsets.  random_state=seed garante reprodutibilidade.

        Divisão padrão: 70% train / 10% val / 20% test.
        - val: seleção de checkpoint e early stopping durante treino.
        - test: avaliação final (NUNCA visto durante treino).

        Se cache_dir for fornecido, salva train.txt, val.txt e test.txt.
        """
        indices = list(range(len(self.words)))

        # Dataset completo (G2PDataset adiciona EOS internamente)
        full_ds = G2PDataset(
            self.words, self.phonemes,
            self.char_vocab, self.phoneme_vocab,
        )

        # Caso especial: 100% test (ex: avaliação de conjunto externo como neologismos)
        # Não há split — todos os pares vão direto para test.
        if test_ratio >= 1.0:
            ds = DataSplit(
                corpus=self, full_dataset=full_ds,
                train_indices=[], val_indices=[], test_indices=indices,
                val_ratio=0.0, test_ratio=1.0, seed=seed,
                cache_tag=cache_tag,
            )
            if cache_dir is not None:
                ds.save_cache(cache_dir)
            return ds

        # Estratificação: colapsar estratos com <2 exemplos
        strata = list(self._strata)
        strata_counts = Counter(strata)
        for i, s in enumerate(strata):
            if strata_counts[s] < 2:
                strata[i] = "__rare__"

        # Se __rare__ ficou com <2 membros, mesclar no estrato mais comum
        strata_counts2 = Counter(strata)
        if "__rare__" in strata_counts2 and strata_counts2["__rare__"] < 2:
            most_common = strata_counts2.most_common(1)[0][0]
            if most_common == "__rare__":
                most_common = strata_counts2.most_common(2)[1][0]
            for i, s in enumerate(strata):
                if s == "__rare__":
                    strata[i] = most_common

        # Passo 1: separar test
        trainval_idx, test_idx = train_test_split(
            indices, test_size=test_ratio, random_state=seed,
            stratify=strata,
        )

        # Passo 2: separar val do train
        # val_ratio é relativo ao total, então ajustar para fração do trainval
        trainval_strata = [strata[i] for i in trainval_idx]
        # Colapsar novamente para o subconjunto
        tv_counts = Counter(trainval_strata)
        for j, s in enumerate(trainval_strata):
            if tv_counts[s] < 2:
                trainval_strata[j] = "__rare__"

        # Se __rare__ ficou com <2 membros, mesclar no estrato mais comum
        tv_counts2 = Counter(trainval_strata)
        if "__rare__" in tv_counts2 and tv_counts2["__rare__"] < 2:
            most_common = tv_counts2.most_common(1)[0][0]
            if most_common == "__rare__":
                most_common = tv_counts2.most_common(2)[1][0]
            for j, s in enumerate(trainval_strata):
                if s == "__rare__":
                    trainval_strata[j] = most_common

        val_fraction = val_ratio / (1.0 - test_ratio)  # e.g. 0.1 / 0.8 = 0.125
        train_idx, val_idx = train_test_split(
            trainval_idx, test_size=val_fraction, random_state=seed,
            stratify=trainval_strata,
        )

        ds = DataSplit(
            corpus=self,
            full_dataset=full_ds,
            train_indices=train_idx,
            val_indices=val_idx,
            test_indices=test_idx,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
            cache_tag=cache_tag,
        )

        if cache_dir is not None:
            ds.save_cache(cache_dir)

        return ds


class DataSplit:
    """
    Resultado de G2PCorpus.split().
    Split 3-way: train / val / test.
    - train: treino dos pesos.
    - val:   seleção de checkpoint e early stopping.
    - test:  avaliação final (NUNCA usado durante treino).
    Fornece datasets, dataloaders, pares brutos, metadados
    e métricas de qualidade da distribuição.
    """

    def __init__(self, corpus: G2PCorpus, full_dataset: G2PDataset,
                 train_indices: list, val_indices: list, test_indices: list,
                 val_ratio: float, test_ratio: float, seed: int,
                 cache_tag: str = ""):
        self.corpus = corpus
        self._full_dataset = full_dataset
        self.train_indices = train_indices
        self.val_indices = val_indices
        self.test_indices = test_indices
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed
        self.cache_tag = cache_tag

    # ---- datasets (Subset padrão do PyTorch) ----

    @property
    def train_dataset(self) -> Subset:
        return Subset(self._full_dataset, self.train_indices)

    @property
    def val_dataset(self) -> Subset:
        return Subset(self._full_dataset, self.val_indices)

    @property
    def test_dataset(self) -> Subset:
        return Subset(self._full_dataset, self.test_indices)

    # ---- dataloaders ----

    def get_dataloaders(self, batch_size: int = 32):
        """Retorna (train_dl, val_dl, test_dl).

        Otimizações (2026-02-23):
        - pin_memory=True: Transfer CPU→GPU mais rápido via pinned memory (CUDA only)
        - num_workers=0: Windows spawn causa reimportação do módulo ao criar workers;
          persistent_workers=True reduz para overhead só na epoch 1, mas ganho steady-state
          é ~1-2s/epoch — insignificante para LSTM sequencial. Mantido em 0 para simplicidade.
          Conclusão: gargalo é LSTM serial em GPU, não CPU/DataLoader.
        """
        use_cuda = torch.cuda.is_available()
        train_dl = DataLoader(
            self.train_dataset, batch_size=batch_size,
            shuffle=True, collate_fn=collate_fn,
            pin_memory=use_cuda, num_workers=0,
        )
        val_dl = DataLoader(
            self.val_dataset, batch_size=batch_size,
            shuffle=False, collate_fn=collate_fn,
            pin_memory=use_cuda, num_workers=0,
        )
        test_dl = DataLoader(
            self.test_dataset, batch_size=batch_size,
            shuffle=False, collate_fn=collate_fn,
            pin_memory=use_cuda, num_workers=0,
        )
        return train_dl, val_dl, test_dl

    # ---- pares brutos (para avaliação / logging) ----

    def train_pairs(self):
        """Retorna (words, phonemes) do split de treino."""
        words = [self.corpus.words[i] for i in self.train_indices]
        phonemes = [self.corpus.phonemes[i] for i in self.train_indices]
        return words, phonemes

    def val_pairs(self):
        """Retorna (words, phonemes) do split de validação."""
        words = [self.corpus.words[i] for i in self.val_indices]
        phonemes = [self.corpus.phonemes[i] for i in self.val_indices]
        return words, phonemes

    def test_pairs(self):
        """Retorna (words, phonemes) do split de teste."""
        words = [self.corpus.words[i] for i in self.test_indices]
        phonemes = [self.corpus.phonemes[i] for i in self.test_indices]
        return words, phonemes

    def cache_suffix(self) -> str:
        """Sufixo estável para nomes de cache, evitando colisões entre configs."""
        train_ratio = max(0.0, 1.0 - self.val_ratio - self.test_ratio)
        train_pct = int(round(train_ratio * 100))
        val_pct = int(round(self.val_ratio * 100))
        test_pct = int(round(self.test_ratio * 100))

        sep_tag = "sep" if self.corpus.keep_syllable_separators else "nosep"

        # Converter grapheme_encoding para string segura para filename
        ge = self.corpus.grapheme_encoding
        if isinstance(ge, dict):
            # Se é um dict com filters, criar um tag descritivo
            gtype = ge.get("type", "raw")
            filters = ge.get("filters", [])
            if "-" in filters:
                ge_tag = f"{gtype}_nohyphen"
            else:
                ge_tag = f"{gtype}_filters"
        else:
            # Se é string, usar como-está
            ge_tag = str(ge)

        return (
            f"_{ge_tag}"
            f"_{sep_tag}"
            f"_{train_pct}-{val_pct}-{test_pct}"
            f"_s{self.seed}"
        )

    def cache_filenames(self) -> dict[str, str]:
        """Retorna nomes de arquivos de cache para train/val/test.

        Se cache_tag for definido (ex: "neologismos"), os nomes ficam
        "neologismos_test_raw_nosep_0-0-100_s42.txt" — mais identificáveis.
        """
        suffix = self.cache_suffix()
        prefix = f"{self.cache_tag}_" if self.cache_tag else ""
        return {
            "train": f"{prefix}train{suffix}.txt",
            "val":   f"{prefix}val{suffix}.txt",
            "test":  f"{prefix}test{suffix}.txt",
        }

    # ---- cache para inspeção humana ----

    def save_cache(self, cache_dir: str | Path):
        """Salva cache em arquivos distintos por configuração para inspeção.

        O nome incorpora encoding, separadores, split e seed, evitando
        sobrescrita silenciosa entre experimentos (ex.: 70/10/20 vs 60/10/30).
        """
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        cache_names = self.cache_filenames()

        for name, indices in [("train", self.train_indices),
                              ("val", self.val_indices),
                              ("test", self.test_indices)]:
            filename = cache_names[name]
            path = cache_dir / filename
            with open(path, "w", encoding="utf-8") as f:
                f.write("# Auto-gerado por G2PCorpus — arquivo de INSPEÇÃO\n")
                f.write(f"# Fonte: {self.corpus.dict_path.name}\n")
                f.write(f"# Checksum: {self.corpus.checksum}\n")
                f.write(f"# Split: seed={self.seed}  val_ratio={self.val_ratio}  test_ratio={self.test_ratio}\n")
                f.write(f"# Grapheme encoding: {self.corpus.grapheme_encoding}\n")
                f.write(f"# Syllable separators: {'keep' if self.corpus.keep_syllable_separators else 'remove'}\n")
                f.write("# Split estratificado por: tonicidade × sílabas × comprimento\n")
                f.write(f"# Total neste split: {len(indices)} palavras\n")
                f.write("#\n")
                for i in indices:
                    f.write(f"{self.corpus.words[i]} {self.corpus.phonemes[i]}\n")
            logger.info("Cache salvo: %s (%d palavras)", path, len(indices))

    # ---- métricas de qualidade da distribuição ----

    def _distribution(self, indices: list, feature_key: str) -> Counter:
        """Conta ocorrências de uma feature nos índices dados."""
        c = Counter()
        for i in indices:
            feat = _extract_features(
                self.corpus.words_raw[i], self.corpus._phonemes_raw[i],
            )
            c[feat[feature_key]] += 1
        return c

    @staticmethod
    def _jsd(p_counts: Counter, q_counts: Counter) -> float:
        """
        Jensen-Shannon Divergence entre duas distribuições (Counter).
        Valor entre 0 (idênticas) e 1 (disjuntas).
        """
        all_keys = set(p_counts) | set(q_counts)
        p_total = sum(p_counts.values())
        q_total = sum(q_counts.values())
        if p_total == 0 or q_total == 0:
            return 1.0

        p = np.array([p_counts.get(k, 0) / p_total for k in all_keys])
        q = np.array([q_counts.get(k, 0) / q_total for k in all_keys])
        m = 0.5 * (p + q)

        # KL(p||m) e KL(q||m) com proteção contra log(0)
        def _kl(a: np.ndarray, b: np.ndarray) -> float:
            mask = a > 0
            return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

        return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)

    def _phoneme_coverage(self, indices: list) -> tuple[int, int, float]:
        """Conta quantos fonemas únicos do corpus todo estão presentes no subset."""
        all_phonemes = set()
        for p in self.corpus.phonemes:
            all_phonemes.update(p.split())
        subset_phonemes = set()
        for i in indices:
            subset_phonemes.update(self.corpus.phonemes[i].split())
        covered = len(subset_phonemes & all_phonemes)
        return covered, len(all_phonemes), covered / max(len(all_phonemes), 1)

    def _bigram_coverage(self, indices: list) -> tuple[int, int, float]:
        """Cobertura de bigramas de fonemas (padrões silábicos)."""
        all_bigrams = set()
        for p in self.corpus.phonemes:
            tokens = p.split()
            for a, b in zip(tokens, tokens[1:]):
                all_bigrams.add((a, b))
        subset_bigrams = set()
        for i in indices:
            tokens = self.corpus.phonemes[i].split()
            for a, b in zip(tokens, tokens[1:]):
                subset_bigrams.add((a, b))
        covered = len(subset_bigrams & all_bigrams)
        return covered, len(all_bigrams), covered / max(len(all_bigrams), 1)

    def quality_report(self) -> dict:
        """
        Computa métricas de qualidade da distribuição do split.

        Métricas incluídas:
          - JSD (Jensen-Shannon Divergence) por feature
          - Chi-squared test (H0: train e test são da mesma distribuição)
          - Cramér's V (tamanho do efeito para qui-quadrado, 0=nenhum, 1=total)
          - Intervalo de confiança 95% para diferença máx de proporção (Wilson)
          - Cobertura de fonemas e bigramas em cada subset (train/val/test)
          - Veredicto geral com nível de confiança
        """
        features = ["stress_type", "syllable_bin", "length_bin", "ratio_bin"]
        report: dict = {
            "jsd": {}, "chi2": {}, "cramers_v": {}, "chi2_pvalue": {},
            "max_delta_ci95": {},
            "train_coverage": {}, "val_coverage": {}, "test_coverage": {},
        }

        n_train = len(self.train_indices)
        n_test = len(self.test_indices)

        for feat in features:
            train_dist = self._distribution(self.train_indices, feat)
            test_dist = self._distribution(self.test_indices, feat)

            # JSD (informational)
            jsd = self._jsd(train_dist, test_dist)
            report["jsd"][feat] = jsd

            # Chi-squared test de homogeneidade
            all_keys = sorted(set(train_dist) | set(test_dist))
            observed = np.array([
                [train_dist.get(k, 0) for k in all_keys],
                [test_dist.get(k, 0) for k in all_keys],
            ])
            # Remover colunas zeradas (scipy exige >0 em marginal)
            col_sums = observed.sum(axis=0)
            observed = observed[:, col_sums > 0]

            if observed.shape[1] >= 2:
                chi2_val, p_val, dof, _ = sp_stats.chi2_contingency(observed)
                # Cramér's V = sqrt(chi2 / (N * (min(r,c)-1)))
                n_total = observed.sum()
                min_dim = min(observed.shape) - 1
                cramers_v = math.sqrt(chi2_val / (n_total * max(min_dim, 1))) \
                    if n_total > 0 else 0.0
            else:
                chi2_val, p_val, _, cramers_v = 0.0, 1.0, 0, 0.0

            report["chi2"][feat] = float(chi2_val)
            report["chi2_pvalue"][feat] = float(p_val)
            report["cramers_v"][feat] = float(cramers_v)

            # Intervalo de confiança 95% para a maior diferença de proporção
            max_delta = 0.0
            max_delta_ci = (0.0, 0.0)
            max_delta_class = ""
            for k in all_keys:
                p_tr = train_dist.get(k, 0) / n_train if n_train else 0
                p_te = test_dist.get(k, 0) / n_test if n_test else 0
                delta = p_tr - p_te
                # Erro padrão da diferença de proporções
                se = math.sqrt(
                    p_tr * (1 - p_tr) / max(n_train, 1)
                    + p_te * (1 - p_te) / max(n_test, 1)
                )
                ci_lo = delta - 1.96 * se
                ci_hi = delta + 1.96 * se
                if abs(delta) > abs(max_delta):
                    max_delta = delta
                    max_delta_ci = (ci_lo, ci_hi)
                    max_delta_class = k

            report["max_delta_ci95"][feat] = {
                "class": max_delta_class,
                "delta": float(max_delta),
                "ci_lo": float(max_delta_ci[0]),
                "ci_hi": float(max_delta_ci[1]),
            }

            # Distribuição percentual para telemetria
            report.setdefault("distributions", {})[feat] = {
                "train": {k: v / n_train for k, v in sorted(train_dist.items())},
                "test": {k: v / n_test for k, v in sorted(test_dist.items())},
            }

        # Cobertura
        for subset_name, indices in [("train_coverage", self.train_indices),
                                      ("val_coverage", self.val_indices),
                                      ("test_coverage", self.test_indices)]:
            cov_n, cov_total, cov_pct = self._phoneme_coverage(indices)
            report[subset_name]["phonemes"] = {
                "covered": cov_n, "total": cov_total, "pct": cov_pct,
            }
            cov_n, cov_total, cov_pct = self._bigram_coverage(indices)
            report[subset_name]["bigrams"] = {
                "covered": cov_n, "total": cov_total, "pct": cov_pct,
            }

        # Veredicto consolidado
        all_pvalues = list(report["chi2_pvalue"].values())
        all_cramers = list(report["cramers_v"].values())
        # Correção de Bonferroni (múltiplos testes)
        bonf_alpha = 0.05 / len(features)
        any_significant = any(p < bonf_alpha for p in all_pvalues)
        max_cramers = max(all_cramers) if all_cramers else 0.0
        min_pvalue = min(all_pvalues) if all_pvalues else 1.0

        # Interpretar
        if not any_significant:
            confidence = "alta"
            verdict = "excelente"
        elif max_cramers < 0.05:
            confidence = "alta"
            verdict = "bom (efeito desprezível)"
        elif max_cramers < 0.10:
            confidence = "média"
            verdict = "aceitável (efeito pequeno)"
        else:
            confidence = "baixa"
            verdict = "RUIM (efeito relevante)"

        report["verdict"] = {
            "quality": verdict,
            "confidence": confidence,
            "any_chi2_significant": any_significant,
            "bonferroni_alpha": bonf_alpha,
            "min_pvalue": min_pvalue,
            "max_cramers_v": max_cramers,
        }

        return report

    def log_quality(self, report: dict | None = None):
        """Loga métricas de qualidade da distribuição com testes estatísticos."""
        report = report or self.quality_report()
        n_total = len(self.train_indices) + len(self.val_indices) + len(self.test_indices)

        logger.info("")
        logger.info("=" * 70)
        logger.info("QUALIDADE DO SPLIT (Estratificação + Testes Estatísticos)")
        logger.info("=" * 70)
        logger.info(
            "Split: %d treino (%.1f%%) | %d val (%.1f%%) | %d teste (%.1f%%) | seed=%d",
            len(self.train_indices), len(self.train_indices) / n_total * 100,
            len(self.val_indices), len(self.val_indices) / n_total * 100,
            len(self.test_indices), len(self.test_indices) / n_total * 100,
            self.seed,
        )
        logger.info("")

        # Tabela de testes estatísticos por feature
        logger.info(
            "  %-16s %10s %10s %10s %10s",
            "Feature", "JSD", "\u03c7\u00b2 p-value", "Cram\u00e9r V", "Qualidade",
        )
        logger.info("  " + "-" * 58)
        for feat in report["jsd"]:
            jsd = report["jsd"][feat]
            pval = report["chi2_pvalue"][feat]
            cv = report["cramers_v"][feat]
            quality = "excelente" if cv < 0.01 else \
                      "bom" if cv < 0.05 else \
                      "aceit\u00e1vel" if cv < 0.10 else "RUIM"
            logger.info(
                "  %-16s %10.6f %10.4f %10.6f %10s",
                feat, jsd, pval, cv, quality,
            )
        logger.info("")

        # IC 95% para maior delta de proporção por feature
        logger.info("Intervalos de confian\u00e7a 95%% (maior \u0394 de propor\u00e7\u00e3o):")
        for feat, ci_data in report["max_delta_ci95"].items():
            zero_in = ci_data["ci_lo"] <= 0 <= ci_data["ci_hi"]
            sig = "n\u00e3o-sig" if zero_in else "SIG!"
            logger.info(
                "  %-16s  classe=%-10s  \u0394=%.4f%%  IC=[%.4f%%, %.4f%%]  %s",
                feat, ci_data["class"],
                ci_data["delta"] * 100,
                ci_data["ci_lo"] * 100,
                ci_data["ci_hi"] * 100,
                sig,
            )
        logger.info("")

        # Distribuição por feature
        for feat in report["distributions"]:
            train_d = report["distributions"][feat]["train"]
            test_d = report["distributions"][feat]["test"]
            logger.info("Distribui\u00e7\u00e3o [%s]:", feat)
            all_keys = sorted(set(train_d) | set(test_d))
            logger.info("  %-14s %8s %8s %8s", "Classe", "Train%", "Test%", "\u0394")
            for k in all_keys:
                tp = train_d.get(k, 0) * 100
                tep = test_d.get(k, 0) * 100
                delta = abs(tp - tep)
                logger.info("  %-14s %7.2f%% %7.2f%% %7.3f%%", k, tp, tep, delta)
            logger.info("")

        # Cobertura
        logger.info("Cobertura (representatividade do subset):")
        for subset_name in ["train_coverage", "val_coverage", "test_coverage"]:
            label = "Treino" if "train" in subset_name else \
                    "Valid" if "val" in subset_name else "Teste"
            cov = report[subset_name]
            logger.info(
                "  %s: fonemas %d/%d (%.1f%%) | bigramas %d/%d (%.1f%%)",
                label,
                cov["phonemes"]["covered"], cov["phonemes"]["total"],
                cov["phonemes"]["pct"] * 100,
                cov["bigrams"]["covered"], cov["bigrams"]["total"],
                cov["bigrams"]["pct"] * 100,
            )
        logger.info("")

        # Veredicto
        v = report["verdict"]
        logger.info("VEREDICTO: %s | Confian\u00e7a: %s", v["quality"], v["confidence"])
        logger.info(
            "  Bonferroni \u03b1=%.4f | min p-value=%.4e | max Cram\u00e9r V=%.6f",
            v["bonferroni_alpha"], v["min_pvalue"], v["max_cramers_v"],
        )
        if v["any_chi2_significant"]:
            logger.info("  \u26a0 Alguma feature tem \u03c7\u00b2 significativo (p < \u03b1/k),")
            logger.info("    mas Cram\u00e9r V=%.6f indica efeito desprez\u00edvel.", v["max_cramers_v"])
        else:
            logger.info("  \u2713 Nenhuma feature tem \u03c7\u00b2 significativo ap\u00f3s Bonferroni.")
        logger.info("=" * 70)
        logger.info("")

    # ---- metadados (salvos junto com o modelo) ----

    def metadata(self, report: dict | None = None) -> dict:
        """Dict para gravar no JSON de metadados do modelo."""
        report = report or self.quality_report()
        return {
            "dict_path": str(self.corpus.dict_path),
            "dict_checksum": self.corpus.checksum,
            "grapheme_encoding": self.corpus.grapheme_encoding,
            "keep_syllable_separators": self.corpus.keep_syllable_separators,
            "total_words": len(self.corpus.words),
            "train_size": len(self.train_indices),
            "val_size": len(self.val_indices),
            "test_size": len(self.test_indices),
            "val_ratio": self.val_ratio,
            "test_ratio": self.test_ratio,
            "seed": self.seed,
            "char_vocab_size": len(self.corpus.char_vocab),
            "phoneme_vocab_size": len(self.corpus.phoneme_vocab),
            "stratified": True,
            "strata_count": len(set(self.corpus._strata)),
            "jsd": report["jsd"],
            "chi2_pvalue": report["chi2_pvalue"],
            "cramers_v": report["cramers_v"],
            "verdict": report["verdict"],
            "train_phoneme_coverage": report["train_coverage"]["phonemes"]["pct"],
            "val_phoneme_coverage": report["val_coverage"]["phonemes"]["pct"],
            "train_bigram_coverage": report["train_coverage"]["bigrams"]["pct"],
            "val_bigram_coverage": report["val_coverage"]["bigrams"]["pct"],
        }


# ---------------------------------------------------------------------------
# Modelo: Encoder-Decoder LSTM com Attention (Bahdanau/additive)
# ---------------------------------------------------------------------------

class Encoder(nn.Module):
    """BiLSTM encoder. Processa a sequência de caracteres."""
    def __init__(self, vocab_size, emb_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            emb_dim, hidden_dim, num_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Projeção do hidden state bidirecional → unidirecional pro decoder
        self.fc_h = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc_c = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, src, src_lengths):
        # src: (batch, src_len)
        embedded = self.dropout(self.embedding(src))  # (batch, src_len, emb_dim)

        # Pack para eficiência com padding
        packed = pack_padded_sequence(
            embedded, src_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, (h, c) = self.lstm(packed)
        encoder_out, _ = pad_packed_sequence(
            packed_out, batch_first=True, total_length=src.shape[1]
        )
        # encoder_out: (batch, src_len, hidden*2) — mantém comprimento original

        # Combina hidden states forward+backward de cada layer
        # h shape: (num_layers*2, batch, hidden) → (num_layers, batch, hidden)
        h = h.view(self.num_layers, 2, -1, self.hidden_dim)
        h = torch.cat([h[:, 0], h[:, 1]], dim=2)  # (num_layers, batch, hidden*2)
        h = torch.tanh(self.fc_h(h))  # (num_layers, batch, hidden)

        c = c.view(self.num_layers, 2, -1, self.hidden_dim)
        c = torch.cat([c[:, 0], c[:, 1]], dim=2)
        c = torch.tanh(self.fc_c(c))

        return encoder_out, (h, c)


class Attention(nn.Module):
    """Bahdanau (additive) attention."""
    def __init__(self, encoder_dim, decoder_dim):
        super().__init__()
        self.W_enc = nn.Linear(encoder_dim, decoder_dim, bias=False)
        self.W_dec = nn.Linear(decoder_dim, decoder_dim, bias=False)
        self.v = nn.Linear(decoder_dim, 1, bias=False)

    def forward(self, decoder_hidden, encoder_out, mask=None):
        # decoder_hidden: (batch, decoder_dim)
        # encoder_out: (batch, src_len, encoder_dim)

        dec_proj = self.W_dec(decoder_hidden).unsqueeze(1)  # (batch, 1, dec_dim)
        enc_proj = self.W_enc(encoder_out)                   # (batch, src_len, dec_dim)

        energy = torch.tanh(dec_proj + enc_proj)  # (batch, src_len, dec_dim)
        score = self.v(energy).squeeze(2)          # (batch, src_len)

        if mask is not None:
            score = score.masked_fill(mask == 0, float('-inf'))

        weights = F.softmax(score, dim=1)          # (batch, src_len)
        context = torch.bmm(weights.unsqueeze(1), encoder_out).squeeze(1)
        # context: (batch, encoder_dim)
        return context, weights


class Decoder(nn.Module):
    """LSTM decoder com attention. Gera fonemas um a um."""
    def __init__(self, vocab_size, emb_dim, hidden_dim, encoder_dim, num_layers, dropout,
                 embedding_type="learned", phoneme_i2p=None):
        super().__init__()
        self.embedding = get_embedding_layer(
            embedding_type=embedding_type,
            vocab_size=vocab_size,
            emb_dim=emb_dim,
            i2p_mapping=phoneme_i2p
        )
        # Store actual embedding dimension (may differ from emb_dim for panphon)
        actual_emb_dim = self.embedding.embedding_dim
        
        self.attention = Attention(encoder_dim, hidden_dim)
        self.lstm = nn.LSTM(
            actual_emb_dim + encoder_dim, hidden_dim, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.output = nn.Linear(hidden_dim + encoder_dim + actual_emb_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward_step(self, input_token, hidden, encoder_out, mask=None):
        """Um passo do decoder."""
        # input_token: (batch, 1)
        embedded = self.dropout(self.embedding(input_token))  # (batch, 1, emb_dim)

        # Attention usando hidden state do layer mais alto
        h_top = hidden[0][-1]  # (batch, hidden_dim)
        context, attn_weights = self.attention(h_top, encoder_out, mask)
        # context: (batch, encoder_dim)

        # Concatena embedding + contexto como input do LSTM
        lstm_input = torch.cat([embedded, context.unsqueeze(1)], dim=2)
        # lstm_input: (batch, 1, emb_dim + encoder_dim)

        lstm_out, hidden = self.lstm(lstm_input, hidden)
        # lstm_out: (batch, 1, hidden_dim)

        # Predição: combina lstm_out + context + embedding
        prediction = torch.cat([
            lstm_out.squeeze(1), context, embedded.squeeze(1)
        ], dim=1)
        prediction = self.output(prediction)  # (batch, vocab_size)

        return prediction, hidden, attn_weights

    def forward(self, encoder_out, hidden, targets, mask=None, teacher_forcing_ratio=0.5):
        """
        Forward pass completo do decoder durante treinamento.
        targets: (batch, target_len) — inclui EOS no final.
        """
        batch_size = targets.shape[0]
        target_len = targets.shape[1]
        vocab_size = self.output.out_features

        outputs = torch.zeros(batch_size, target_len, vocab_size, device=targets.device)

        # Primeiro input: EOS como token inicial (start-of-sequence)
        # Reutilizamos EOS como SOS — convenção comum em modelos pequenos
        input_token = torch.full((batch_size, 1), PhonemeVocab.EOS_IDX,
                                 dtype=torch.long, device=targets.device)

        for t in range(target_len):
            prediction, hidden, _ = self.forward_step(input_token, hidden, encoder_out, mask)
            outputs[:, t] = prediction

            # Teacher forcing: usa target real ou predição do modelo
            if torch.rand(1).item() < teacher_forcing_ratio:
                input_token = targets[:, t].unsqueeze(1)
            else:
                input_token = prediction.argmax(dim=1).unsqueeze(1)

        return outputs


class G2PLSTMModel(nn.Module):
    """
    Encoder-Decoder LSTM com Attention para G2P.
    
    Tokens especiais (internos ao modelo, não existem nos dados):
    - PAD (0): padding para batches. Embedding fixo zero. Ignorado na loss.
    - UNK (1): token desconhecado. Embedding aprendido.
    - EOS (2): fim de sequência. Embedding aprendido. O decoder para quando
               prediz EOS. Adicionado ao target pelo G2PDataset automaticamente.
    """
    def __init__(self, char_vocab_size, phoneme_vocab_size,
                 emb_dim=128, hidden_dim=256, num_layers=2, dropout=0.2,
                 embedding_type="learned", phoneme_i2p=None):
        super().__init__()
        self.encoder = Encoder(char_vocab_size, emb_dim, hidden_dim, num_layers, dropout)
        self.decoder = Decoder(
            phoneme_vocab_size, emb_dim, hidden_dim,
            encoder_dim=hidden_dim * 2,  # bidirecional
            num_layers=num_layers, dropout=dropout,
            embedding_type=embedding_type,
            phoneme_i2p=phoneme_i2p
        )
        self.hidden_dim = hidden_dim

    def forward(self, src, src_lengths, targets, teacher_forcing_ratio=0.5):
        """
        Treino: recebe src (chars) e targets (phonemes + EOS).
        Retorna logits (batch, target_len, vocab_size).
        """
        # Máscara para ignorar PAD no attention
        mask = (src != 0)  # (batch, src_len)

        encoder_out, hidden = self.encoder(src, src_lengths)
        outputs = self.decoder(encoder_out, hidden, targets, mask, teacher_forcing_ratio)
        return outputs

    def predict(self, src, src_lengths, max_len=50):
        """
        Inferência: gera fonemas até EOS ou max_len.
        Retorna lista de listas de índices (sem EOS/PAD).
        """
        self.eval()
        with torch.no_grad():
            mask = (src != 0)
            encoder_out, hidden = self.encoder(src, src_lengths)

            batch_size = src.shape[0]
            input_token = torch.full((batch_size, 1), PhonemeVocab.EOS_IDX,
                                     dtype=torch.long, device=src.device)

            all_predictions = [[] for _ in range(batch_size)]
            finished = [False] * batch_size

            for _ in range(max_len):
                prediction, hidden, _ = self.decoder.forward_step(
                    input_token, hidden, encoder_out, mask
                )
                pred_idx = prediction.argmax(dim=1)  # (batch,)
                input_token = pred_idx.unsqueeze(1)

                for b in range(batch_size):
                    if not finished[b]:
                        idx = pred_idx[b].item()
                        if idx == PhonemeVocab.EOS_IDX:
                            finished[b] = True
                        elif idx != PhonemeVocab.PAD_IDX:
                            all_predictions[b].append(idx)

                if all(finished):
                    break

            return all_predictions
