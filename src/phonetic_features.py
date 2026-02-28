# -*- coding: utf-8 -*-
"""
Phonetic Feature Space — PanPhon integration for FG2P.

Mapeia PhonemeVocab → espaço articulatório via PanPhon (COLING 2016).
Fornece distâncias fonéticas, classificação de erros graduados (A/B/C/D),
e escalas articulatórias contínuas (trajetórias).

Este módulo é 100% independente. Não modifica nenhum código existente.
Pode ser importado opcionalmente por inference.py para métricas avançadas.

Uso:
    from g2p import PhonemeVocab
    from phonetic_features import PhoneticSpace
    
    phoneme_vocab = PhonemeVocab()
    # ... popular vocab ...
    
    ps = PhoneticSpace(phoneme_vocab)
    dist = ps.distance(idx_a, idx_b)  # [0,1]
    error_class = ps.classify_error(pred_idx, ref_idx)  # 'A'/'B'/'C'/'D'
    trajectory = ps.articulatory_scale('altura')  # {idx: float [0,1]}

Compatibilidade:
    - Python 3.13+
    - PanPhon 0.22.2+ (pip install panphon)
    - Numpy (já presente no projeto)

Autores: FG2P Project, 2026
Licença: MIT
"""

import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np

# WINDOWS FIX: PanPhon lê arquivos CSV internos (ipa_all.csv, feature_weights.csv)
# via files("panphon").joinpath(fn).open() SEM encoding='utf-8'.
# No Python 3.10+, open() sem encoding usa io.text_encoding(None) → "locale",
# que é resolvido no nível C (GetACP() → cp1252 no Windows).
# locale.getpreferredencoding monkey-patch NÃO funciona porque o C bypassa o Python.
# Solução: monkey-patch _read_bases e _read_weights do PanPhon para forçar encoding='utf-8'.
_PANPHON_AVAILABLE = False
_import_error = None

try:
    import panphon
    import panphon.distance
    import panphon.featuretable

    if sys.platform == 'win32':
        from importlib.resources import files as _pkg_files
        import pandas as _pd
        from panphon.segment import Segment as _Segment

        _original_normalize = panphon.featuretable.FeatureTable.normalize

        def _patched_read_bases(self, fn, weights):
            """_read_bases com encoding='utf-8' explícito (Windows fix)."""
            spec_to_int = {"+": 1, "0": 0, "-": -1}
            with _pkg_files("panphon").joinpath(fn).open(encoding='utf-8') as f:
                df = _pd.read_csv(f)
            df["ipa"] = df["ipa"].apply(self.normalize)
            feature_names = list(df.columns[1:])
            df[feature_names] = df[feature_names].map(lambda x: spec_to_int[x])
            segments = [
                (row["ipa"], _Segment(feature_names, row[1:].to_dict(), weights=weights))
                for (_, row) in df.iterrows()
            ]
            seg_dict = dict(segments)
            return segments, seg_dict, feature_names

        def _patched_read_weights(self, weights_fn):
            """_read_weights com encoding='utf-8' explícito (Windows fix)."""
            with _pkg_files('panphon').joinpath(weights_fn).open(encoding='utf-8') as f:
                df = _pd.read_csv(f)
            return df.iloc[0].astype(float).tolist()

        panphon.featuretable.FeatureTable._read_bases = _patched_read_bases
        panphon.featuretable.FeatureTable._read_weights = _patched_read_weights
        logger_init = logging.getLogger("phonetic_features")
        logger_init.debug("Windows: PanPhon _read_bases/_read_weights patched com encoding='utf-8'")

    _PANPHON_AVAILABLE = True
except ImportError as e:
    _import_error = str(e)
except Exception as e:
    _import_error = f"Unexpected error: {str(e)}"

logger = logging.getLogger("phonetic_features")


class PhoneticSpace:
    """
    Espaço fonético baseado em features articulatórias (PanPhon).
    
    Para cada índice do PhonemeVocab:
      - PAD(0), UNK(1), EOS(2) → vetor zero [24]
      - Fonemas IPA → PanPhon feature vector [24] com valores {-1, 0, +1}
      - 'g' ASCII → 'ɡ' IPA (U+0261) automaticamente
      - 'ˈ' (stress) → vetor zero (suprassegmental, sem features)
    
    Features PanPhon (24):
        syl, son, cons, cont, delrel, lat, nas, strid, voi, sg, cg,
        ant, cor, distr, lab, hi, lo, back, round, velaric, tense,
        long, hitone, hireg
    
    Attributes:
        vocab_size (int): Tamanho do PhonemeVocab (tipicamente 43)
        feature_dim (int): Dimensão do vetor de features (24)
        _features (np.ndarray): Matriz [vocab_size × 24] de features
        _distance_matrix (np.ndarray): Matriz [vocab_size × vocab_size] de distâncias
    """
    
    def __init__(self, phoneme_vocab):
        """
        Constrói espaço fonético para um PhonemeVocab.
        
        Args:
            phoneme_vocab: Instância de PhonemeVocab (de g2p.py)
        
        Raises:
            ImportError: Se PanPhon não estiver instalado
        """
        if not _PANPHON_AVAILABLE:
            raise ImportError(
                "PanPhon não está instalado. Execute: pip install panphon\n"
                "Ou desabilite métricas fonéticas avançadas."
            )
        
        self.vocab_size = len(phoneme_vocab)
        self.feature_dim = 24
        self._phoneme_vocab = phoneme_vocab
        
        # Inicializar PanPhon
        self._ft = panphon.featuretable.FeatureTable()
        self._distance = panphon.distance.Distance()
        
        # Construir matriz de features [vocab_size × 24]
        self._features = self._build_feature_matrix()
        
        # Pré-computar matriz de distâncias [vocab_size × vocab_size]
        self._distance_matrix = self._compute_distance_matrix()
        
        logger.info(
            "PhoneticSpace inicializado: %d fonemas × %d features",
            self.vocab_size, self.feature_dim
        )
    
    def _normalize_phoneme(self, phoneme: str) -> str:
        """
        Normaliza fonema para IPA padrão do PanPhon.
        
        Correções:
          - 'g' ASCII → 'ɡ' IPA (U+0261)
          - Outros já estão em IPA no pt-br.tsv
        
        Args:
            phoneme: String do fonema (ex: 'a', 'e', 'ɛ', 'g')
        
        Returns:
            Fonema normalizado para IPA
        """
        if phoneme == 'g':
            return 'ɡ'  # U+0261 Latin Small Letter Script G
        return phoneme
    
    def _get_panphon_vector(self, phoneme: str) -> Optional[np.ndarray]:
        """
        Obtém vetor de features PanPhon para um fonema.
        
        Args:
            phoneme: String IPA do fonema
        
        Returns:
            Array numpy [24] com valores {-1, 0, +1}, ou None se não reconhecido
        """
        phoneme = self._normalize_phoneme(phoneme)
        
        # Tokens especiais, suprassegmentais e estruturais → vetor zero
        # '.' e 'ˈ' são símbolos estruturais (separador silábico e stress),
        # não segmentos fonéticos. PanPhon não os reconhece e imprimiria avisos
        # internos (print nu) se passados para word_fts(). Early-return evita isso.
        if phoneme in ('<PAD>', '<UNK>', '<EOS>', 'ˈ', '.'):
            logger.debug("Símbolo estrutural '%s' → vetor zero (comportamento esperado).", phoneme)
            return np.zeros(self.feature_dim, dtype=np.int8)
        
        # Buscar no PanPhon
        try:
            segments = self._ft.word_fts(phoneme)
            
            if len(segments) == 0:
                logger.warning(
                    "Fonema '%s' não reconhecido pelo PanPhon. Usando vetor zero.",
                    phoneme
                )
                return np.zeros(self.feature_dim, dtype=np.int8)
            
            # Pegar primeiro segmento (palavra = 1 fonema)
            seg = segments[0]
            vec = np.array(seg.numeric(), dtype=np.int8)
            return vec
        
        except Exception as e:
            logger.warning(
                "Erro ao processar fonema '%s' no PanPhon: %s. Usando vetor zero.",
                phoneme, e
            )
            return np.zeros(self.feature_dim, dtype=np.int8)
    
    def _build_feature_matrix(self) -> np.ndarray:
        """
        Constrói matriz de features [vocab_size × 24].
        
        Returns:
            Matriz numpy com features de todos os fonemas do vocabulário
        """
        matrix = np.zeros((self.vocab_size, self.feature_dim), dtype=np.int8)
        
        for idx in range(self.vocab_size):
            phoneme = self._phoneme_vocab.i2p.get(idx, '<UNK>')
            vec = self._get_panphon_vector(phoneme)
            if vec is not None:
                matrix[idx] = vec
        
        return matrix
    
    def _compute_distance_matrix(self) -> np.ndarray:
        """
        Pré-computa matriz de distâncias Hamming normalizadas [vocab_size × vocab_size].
        
        Distância Hamming normalizada = (nº features diferentes) / 24
        Escala: [0, 1]
          - 0.0 = fonemas idênticos
          - ~0.04 = 1 feature de diferença (e↔ɛ, p↔b)
          - ~0.38 = vogal vs consoante (a↔k)
        
        Returns:
            Matriz simétrica [V × V] de distâncias normalizadas
        """
        V = self.vocab_size
        dist_matrix = np.zeros((V, V), dtype=np.float32)
        
        for i in range(V):
            for j in range(i+1, V):
                # Hamming distance = soma de features diferentes
                hamming = np.sum(self._features[i] != self._features[j])
                # Normalizar por 24 (ou menos se alguns features são 0 em ambos)
                # Para simplificar, usar 24 fixo (consistente com PanPhon)
                normalized = hamming / 24.0
                dist_matrix[i, j] = normalized
                dist_matrix[j, i] = normalized  # simetria

        # --- Post-hoc: distâncias customizadas para símbolos estruturais ---
        # Justificativa: '.' e 'ˈ' têm vetor zero em _get_panphon_vector(), então
        # hamming(., ˈ) = 0 → distance = 0.0. Override para distância máxima (1.0).
        structural_phonemes = {'.', 'ˈ'}
        for idx in range(V):
            phoneme = self._phoneme_vocab.i2p.get(idx, '<UNK>')
            if phoneme not in structural_phonemes:
                continue
            for jdx in range(V):
                if idx == jdx:
                    continue  # identidade: 0.0 (não tocar)
                dist_matrix[idx, jdx] = 1.0
                dist_matrix[jdx, idx] = 1.0
        # --- fim overrides estruturais ---

        return dist_matrix
    
    def distance(self, idx_a: int, idx_b: int) -> float:
        """
        Distância fonética entre dois índices do vocabulário.
        
        Args:
            idx_a: Índice do primeiro fonema
            idx_b: Índice do segundo fonema
        
        Returns:
            Distância normalizada [0, 1]
              - 0.0 = idênticos
              - 0.04 = 1 feature diferente
              - 0.38 = muito distantes (vogal vs consoante)
        """
        if idx_a >= self.vocab_size or idx_b >= self.vocab_size:
            return 1.0  # máxima distância para índices inválidos
        
        return float(self._distance_matrix[idx_a, idx_b])
    
    def distance_matrix(self) -> np.ndarray:
        """
        Retorna a matriz completa de distâncias [vocab_size × vocab_size].
        
        Útil para:
          - Visualização (heatmap)
          - Loss ponderada (weights = distance_matrix[target])
          - Análise de confusão fonética
        
        Returns:
            Matriz numpy simétrica [V × V]
        """
        return self._distance_matrix.copy()
    
    def classify_error(self, pred_idx: int, ref_idx: int) -> str:
        """
        Classifica erro entre predição e referência em 4 classes graduadas.
        
        Classes:
          - 'A': Exato (distância = 0)
          - 'B': Quase-idêntico (distância ≤ 0.05, ~1 feature)
          - 'C': Mesma família (distância ≤ 0.15, 2-3 features)
          - 'D': Distante (distância > 0.15)
        
        Args:
            pred_idx: Índice do fonema predito
            ref_idx: Índice do fonema de referência
        
        Returns:
            String 'A', 'B', 'C' ou 'D'
        """
        if pred_idx == ref_idx:
            return 'A'  # exato
        
        dist = self.distance(pred_idx, ref_idx)
        
        if dist <= 0.05:  # ~1 feature de diferença
            return 'B'  # quase-idêntico (e↔ɛ, o↔ɔ, p↔b)
        elif dist <= 0.15:  # 2-3 features
            return 'C'  # mesma família (a↔e, s↔z)
        else:
            return 'D'  # distante (a↔k, vogal↔consoante)
    
    def feature_vector(self, idx: int) -> np.ndarray:
        """
        Retorna o vetor de features PanPhon [24] para um índice do vocabulário.
        
        Args:
            idx: Índice do fonema no PhonemeVocab
        
        Returns:
            Array numpy [24] com valores {-1, 0, +1}
        """
        if idx >= self.vocab_size:
            return np.zeros(self.feature_dim, dtype=np.int8)
        return self._features[idx].copy()
    
    def feature_matrix(self) -> np.ndarray:
        """
        Retorna a matriz completa de features [vocab_size × 24].
        
        Útil para:
          - Inicialização de embedding: `emb.weight.data = torch.from_numpy(ps.feature_matrix())`
          - PCA/UMAP para redução de dimensionalidade
          - Análise de clusters fonéticos
        
        Returns:
            Matriz numpy [V × 24]
        """
        return self._features.copy()
    
    def articulatory_scale(self, axis: str) -> dict[int, float]:
        """
        TRAJETÓRIA ARTICULATÓRIA — mapeia índices do vocabulário para
        posições contínuas [0, 1] em um eixo articulatório específico.
        
        Esta é a ideia ORIGINAL proposta pelo usuário: em vez de features
        binárias independentes, modelar o espaço fonético como um grafo
        onde cada eixo articulatório (lábios, língua, etc.) define uma
        trajetória contínua entre fonemas.
        
        Eixos disponíveis:
          'altura'      → Altura da língua: a(0.0) → e(0.5) → i(1.0)
          'posicao'     → Posição antero-posterior: i(0.0) → a(0.5) → u(1.0)
          'arredond'    → Arredondamento labial: i(0.0) → o(0.5) → u(1.0)
          'nasalidade'  → Nasalização: oral(0.0) → nasal(1.0)
          'vozeamento'  → Vozeamento (consoantes): surda(0.0) → sonora(1.0)
          'modo'        → Modo articulatório: oclusiva(0.0) → fricativa(0.5) → soante(1.0)
        
        Implementação:
          - Usa as features PanPhon 'hi', 'lo', 'back', 'round', 'nas', 'voi'
          - Mapeia valores {-1, 0, +1} → [0, 1] contínuo
          - Vogais: usa features de altura/posição
          - Consoantes: usa features de modo/ponto
          - Tokens especiais (PAD, UNK, EOS): 0.5 (neutro)
        
        Args:
            axis: Nome do eixo articulatório
        
        Returns:
            Dicionário {idx → posição [0,1]} para cada índice do vocabulário
        
        Raises:
            ValueError: Se o eixo não for reconhecido
        
        Example:
            >>> ps = PhoneticSpace(phoneme_vocab)
            >>> altura = ps.articulatory_scale('altura')
            >>> # altura[idx_a] = 0.0 (baixa)
            >>> # altura[idx_e] = 0.5 (média)
            >>> # altura[idx_i] = 1.0 (alta)
            >>> # Distância no eixo: |altura[idx_a] - altura[idx_i]| = 1.0
        """
        axis = axis.lower()
        
        # Mapa de features PanPhon (índices na matriz)
        feature_names = [
            'syl', 'son', 'cons', 'cont', 'delrel', 'lat', 'nas', 'strid',
            'voi', 'sg', 'cg', 'ant', 'cor', 'distr', 'lab', 'hi',
            'lo', 'back', 'round', 'velaric', 'tense', 'long', 'hitone', 'hireg'
        ]
        
        def get_feature_idx(name: str) -> int:
            try:
                return feature_names.index(name)
            except ValueError:
                raise ValueError(f"Feature '{name}' não existe no PanPhon")
        
        scale = {}
        
        if axis == 'altura':
            # Altura da língua: combinação de 'hi' e 'lo'
            # hi=+1, lo=-1 → alta (i, u) → 1.0
            # hi=-1, lo=+1 → baixa (a) → 0.0
            # hi=0, lo=0 → média (e, o) → 0.5
            hi_idx = get_feature_idx('hi')
            lo_idx = get_feature_idx('lo')
            
            for idx in range(self.vocab_size):
                hi_val = self._features[idx, hi_idx]
                lo_val = self._features[idx, lo_idx]
                
                # Mapeamento: lo=+1 → 0.0, neutro → 0.5, hi=+1 → 1.0
                if lo_val == 1:
                    scale[idx] = 0.0  # baixa
                elif hi_val == 1:
                    scale[idx] = 1.0  # alta
                elif lo_val == -1 and hi_val == -1:
                    scale[idx] = 0.5  # média (e, o)
                else:
                    scale[idx] = 0.5  # neutro (consoantes, tokens especiais)
        
        elif axis == 'posicao':
            # Posição antero-posterior: 'back'
            # back=-1 → anterior (i, e) → 0.0
            # back=0 → central (a, ə) → 0.5
            # back=+1 → posterior (u, o) → 1.0
            back_idx = get_feature_idx('back')
            
            for idx in range(self.vocab_size):
                back_val = self._features[idx, back_idx]
                scale[idx] = (back_val + 1) / 2.0  # {-1,0,+1} → {0.0, 0.5, 1.0}
        
        elif axis == 'arredond':
            # Arredondamento labial: 'round'
            # round=-1 → não-arredondado (i, e, a) → 0.0
            # round=+1 → arredondado (u, o) → 1.0
            round_idx = get_feature_idx('round')
            
            for idx in range(self.vocab_size):
                round_val = self._features[idx, round_idx]
                scale[idx] = (round_val + 1) / 2.0
        
        elif axis == 'nasalidade':
            # Nasalização: 'nas'
            # nas=-1 → oral → 0.0
            # nas=+1 → nasal → 1.0
            nas_idx = get_feature_idx('nas')
            
            for idx in range(self.vocab_size):
                nas_val = self._features[idx, nas_idx]
                scale[idx] = (nas_val + 1) / 2.0
        
        elif axis == 'vozeamento':
            # Vozeamento: 'voi'
            # voi=-1 → surda (p, t, k, f, s) → 0.0
            # voi=+1 → sonora (b, d, g, v, z) → 1.0
            voi_idx = get_feature_idx('voi')
            
            for idx in range(self.vocab_size):
                voi_val = self._features[idx, voi_idx]
                scale[idx] = (voi_val + 1) / 2.0
        
        elif axis == 'modo':
            # Modo articulatório: combinação de 'son' (soante) e 'cont' (contínuo)
            # son=-1, cont=-1 → oclusiva (p, t, k) → 0.0
            # son=-1, cont=+1 → fricativa (f, s) → 0.33
            # son=+1, cont=-1 → nasal/lateral (m, n, l) → 0.66
            # son=+1, cont=+1 → vogal/aproximante → 1.0
            son_idx = get_feature_idx('son')
            cont_idx = get_feature_idx('cont')
            
            for idx in range(self.vocab_size):
                son_val = self._features[idx, son_idx]
                cont_val = self._features[idx, cont_idx]
                
                if son_val == -1 and cont_val == -1:
                    scale[idx] = 0.0  # oclusiva
                elif son_val == -1 and cont_val == 1:
                    scale[idx] = 0.33  # fricativa
                elif son_val == 1 and cont_val == -1:
                    scale[idx] = 0.66  # nasal/lateral
                elif son_val == 1 and cont_val == 1:
                    scale[idx] = 1.0  # vogal/aproximante
                else:
                    scale[idx] = 0.5  # neutro
        
        else:
            raise ValueError(
                f"Eixo '{axis}' não reconhecido. "
                f"Opções: altura, posicao, arredond, nasalidade, vozeamento, modo"
            )
        
        return scale
    
    def graph_distance(self, idx_a: int, idx_b: int, axes: list[str]) -> float:
        """
        Distância entre fonemas como caminho no grafo articulatório.
        
        Esta é a extensão da ideia de trajetórias: em vez de usar Hamming
        sobre todas as features, calcula a distância como a soma dos
        deslocamentos em cada eixo articulatório.
        
        Exemplo:
          a → o: move no eixo 'altura' (0.0→0.5) e 'arredond' (0.0→1.0)
          Distância = |0.5| + |1.0| = 1.5 (normalizada por nº de eixos)
        
        Args:
            idx_a: Índice do fonema de origem
            idx_b: Índice do fonema de destino
            axes: Lista de eixos a considerar (ex: ['altura', 'arredond'])
        
        Returns:
            Distância normalizada pelo número de eixos [0, N]
              onde N = len(axes)
        """
        total_distance = 0.0
        
        for axis in axes:
            scale = self.articulatory_scale(axis)
            pos_a = scale.get(idx_a, 0.5)
            pos_b = scale.get(idx_b, 0.5)
            total_distance += abs(pos_a - pos_b)
        
        # Normalizar pela quantidade de eixos para [0, 1] médio
        return total_distance / max(len(axes), 1)
    
    def get_phoneme_name(self, idx: int) -> str:
        """
        Retorna o nome legível do fonema para um índice.
        
        Args:
            idx: Índice no PhonemeVocab
        
        Returns:
            String do fonema (ex: 'a', 'ɛ', '<PAD>')
        """
        return self._phoneme_vocab.i2p.get(idx, '<UNK>')
    
    def summary(self) -> dict:
        """
        Retorna resumo estatístico do espaço fonético.
        
        Returns:
            Dict com estatísticas:
              - vocab_size: tamanho do vocabulário
              - feature_dim: dimensão dos vetores de features
              - avg_distance: distância média entre todos os pares
              - max_distance: distância máxima observada
              - min_distance_nonzero: menor distância > 0
        """
        # Triangular superior (sem diagonal)
        triu_indices = np.triu_indices(self.vocab_size, k=1)
        distances = self._distance_matrix[triu_indices]
        
        nonzero_distances = distances[distances > 0]
        
        return {
            'vocab_size': self.vocab_size,
            'feature_dim': self.feature_dim,
            'avg_distance': float(np.mean(distances)),
            'max_distance': float(np.max(distances)),
            'min_distance_nonzero': float(np.min(nonzero_distances)) if len(nonzero_distances) > 0 else 0.0,
            'panphon_version': panphon.__version__ if hasattr(panphon, '__version__') else 'unknown',
        }


# ---------------------------------------------------------------------------
# Funções utilitárias para uso externo
# ---------------------------------------------------------------------------

def load_phoneme_map(map_path: Optional[Path] = None) -> dict:
    """
    Carrega mapa de normalização de fonemas para PanPhon.
    
    O arquivo phoneme_map.json contém mapeamento de 8-9 fonemas 
    português-brasileiros que não existem em PanPhon para seus
    equivalentes (ou skip markers).
    
    Args:
        map_path: Caminho ao phoneme_map.json. Se None, procura em data/
    
    Returns:
        Dict: {'ã': 'a', 'õ': 'o', ..., 'ˈ': '<SKIP>'}
        Se arquivo não encontrado, retorna dict vazio (sem normalização)
    """
    if map_path is None:
        map_path = Path(__file__).parent.parent / "data" / "phoneme_map.json"
    
    if not map_path.exists():
        logger.warning(f"phoneme_map.json não encontrado em {map_path}. " +
                      "Graduadas metrics funcionarão mas podem ter erros para fonemas ausentes em PanPhon.")
        return {}
    
    try:
        import json
        with open(map_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        phoneme_map = data.get('map_to_panphon', {})
        logger.debug(f"Loaded phoneme_map: {len(phoneme_map)} mapeamentos")
        return phoneme_map
    except Exception as e:
        logger.error(f"Erro ao carregar phoneme_map.json: {e}")
        return {}


def graduated_metrics(predictions: list[list[int]], 
                     references: list[list[int]],
                     phonetic_space: PhoneticSpace,
                     phoneme_vocab: Optional['PhonemeVocab'] = None,
                     phoneme_map: Optional[dict] = None,
                     words: Optional[list[str]] = None) -> dict:
    """
    Calcula métricas graduadas (WER-graduado, PER-ponderado, distribuição A/B/C/D).
    
    Esta função é chamada por analyze_errors.py para métricas avançadas.
    Suporta mapeamento de fonemas para PanPhon via phoneme_map.
    
    Args:
        predictions: Lista de sequências predichas (índices)
        references: Lista de sequências de referência (índices)
        phonetic_space: Instância de PhoneticSpace
        phoneme_vocab: (Opcional) PhonemeVocab para resolver nomes. Usar se phoneme_map fornecido.
        phoneme_map: (Opcional) Dict de normalização {'ã': 'a', ...}. 
                     Carrega de data/phoneme_map.json se não fornecido.
        words: (Opcional) Lista de palavras correspondentes, para enriquecer graduated_word_scores.
    
    Returns:
        Dict com:
          - wer_graduated: WER considerando acertos parciais (classes A/B/C)
          - per_weighted: PER ponderado por distância fonética
          - error_distribution: Counter de classes {'A': n, 'B': m, ...}
          - class_proportions: % de cada classe
          - phonetic_confusion: Top-10 confusões com distâncias
          - graduated_word_scores: Lista de dicts {word, pred, ref, score, worst_class}
          - total_phonemes: Total de fonemas avaliados
          - total_words: Total de palavras
    """
    from collections import Counter
    
    # Carregar mapa se não fornecido
    if phoneme_map is None:
        phoneme_map = load_phoneme_map()
    
    total_phonemes = 0
    total_words = len(predictions)
    
    error_classes = Counter()
    weighted_error = 0.0
    graduated_word_scores = []
    
    phonetic_confusion = Counter()  # (pred_phoneme, ref_phoneme) → count
    
    class_order = {'A': 0, 'B': 1, 'C': 2, 'D': 3}  # Para determinar pior classe
    
    for word_idx, (pred_seq, ref_seq) in enumerate(zip(predictions, references)):
        # Alinhar sequências (simplificado: assumir mesmo tamanho ou truncar)
        min_len = min(len(pred_seq), len(ref_seq))
        max_len = max(len(pred_seq), len(ref_seq))
        
        word_score = 0.0
        word_total = max_len
        worst_class = 'A'
        
        for i in range(max_len):
            pred_idx = pred_seq[i] if i < len(pred_seq) else 0  # PAD
            ref_idx = ref_seq[i] if i < len(ref_seq) else 0
            
            total_phonemes += 1
            
            # Obter nomes para mapeamento (se vocab fornecido)
            pred_name = phoneme_vocab.i2p[pred_idx] if phoneme_vocab else phonetic_space.get_phoneme_name(pred_idx)
            ref_name = phoneme_vocab.i2p[ref_idx] if phoneme_vocab else phonetic_space.get_phoneme_name(ref_idx)
            
            # Aplicar mapa de normalização
            pred_mapped = phoneme_map.get(pred_name, pred_name)
            ref_mapped = phoneme_map.get(ref_name, ref_name)
            
            # Classificar erro (com fallback para <SKIP>)
            if pred_mapped == '<SKIP>' and ref_mapped == '<SKIP>':
                error_class = 'A'  # ambos ignorados = exato
            elif pred_mapped == '<SKIP>' or ref_mapped == '<SKIP>':
                error_class = 'D'  # um ignorado = erro máximo
            else:
                error_class = phonetic_space.classify_error(pred_idx, ref_idx)
            
            error_classes[error_class] += 1
            
            # Atualizar pior classe da palavra
            if class_order.get(error_class, 3) > class_order.get(worst_class, 0):
                worst_class = error_class
            
            # Score graduado por classe
            class_weights = {'A': 1.0, 'B': 0.75, 'C': 0.25, 'D': 0.0}
            score = class_weights[error_class]
            word_score += score
            
            # Erro ponderado por distância (para PER-ponderado)
            # Usar fallback se fonemas não estão em PanPhon
            try:
                dist = phonetic_space.distance(pred_idx, ref_idx)
            except:
                dist = 0.0 if pred_idx == ref_idx else 1.0
            weighted_error += dist
            
            # Registrar confusão (se erro)
            if error_class != 'A':
                phonetic_confusion[(ref_name, pred_name)] += 1
        
        # Score da palavra: média dos fonemas
        word_final_score = word_score / word_total if word_total > 0 else 0.0
        
        # Construir registro enriquecido
        pred_phonemes = [phoneme_vocab.i2p[idx] for idx in pred_seq] if phoneme_vocab else [str(idx) for idx in pred_seq]
        ref_phonemes = [phoneme_vocab.i2p[idx] for idx in ref_seq] if phoneme_vocab else [str(idx) for idx in ref_seq]
        
        word_record = {
            'word': words[word_idx] if words and word_idx < len(words) else f'word_{word_idx}',
            'pred': pred_phonemes,
            'ref': ref_phonemes,
            'score': word_final_score,
            'worst_class': worst_class,
        }
        graduated_word_scores.append(word_record)
    
    # WER graduado: % de palavras com score perfeito (1.0)
    # Ou média dos scores (mais generoso)
    scores_array = [ws['score'] for ws in graduated_word_scores]
    wer_graduated = 100.0 * (1.0 - np.mean(scores_array)) if scores_array else 0.0
    
    # PER ponderado: erro médio (0.0 = perfeito, 1.0 = máximo erro)
    per_weighted = 100.0 * (weighted_error / total_phonemes if total_phonemes > 0 else 0.0)
    
    # Proporções de classes
    total_class = sum(error_classes.values())
    class_proportions = {
        cls: (count / total_class * 100.0 if total_class > 0 else 0.0)
        for cls, count in error_classes.items()
    }
    
    # Top confusões com distâncias
    top_confusion = []
    for (ref_ph, pred_ph), count in phonetic_confusion.most_common(10):
        ref_idx = None
        pred_idx = None
        # Buscar índices (hack: iterar vocab)
        for idx in range(phonetic_space.vocab_size):
            if phonetic_space.get_phoneme_name(idx) == ref_ph:
                ref_idx = idx
            if phonetic_space.get_phoneme_name(idx) == pred_ph:
                pred_idx = idx
        
        if ref_idx is not None and pred_idx is not None:
            dist = phonetic_space.distance(ref_idx, pred_idx)
            error_class = phonetic_space.classify_error(pred_idx, ref_idx)
        else:
            dist = 0.0 if ref_ph == pred_ph else 1.0
            error_class = 'A' if dist == 0.0 else 'D'
        
        top_confusion.append({
            'ref': ref_ph,
            'pred': pred_ph,
            'count': count,
            'distance': dist,
            'class': error_class,
        })
    
    return {
        'wer_graduated': wer_graduated,
        'per_weighted': per_weighted,
        'error_distribution': dict(error_classes),
        'class_proportions': class_proportions,
        'phonetic_confusion': top_confusion,
        'graduated_word_scores': graduated_word_scores,
        'total_phonemes': total_phonemes,
        'total_words': total_words,
    }


# ---------------------------------------------------------------------------
# Auto-teste (executável como script)
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    # Este bloco só roda se executar diretamente: python src/phonetic_features.py
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    
    from g2p import PhonemeVocab
    
    print("=== Teste do PhoneticSpace ===\n")
    
    # Criar vocabulário de teste
    vocab = PhonemeVocab()
    test_phonemes = ['a', 'e', 'ɛ', 'i', 'o', 'ɔ', 'u', 'ã', 'p', 'b', 't', 'd', 'k', 'g', 's', 'z']
    vocab.add(test_phonemes)
    
    print(f"Vocabulário: {len(vocab)} fonemas")
    print(f"  {', '.join(vocab.i2p[i] for i in range(len(vocab)))}\n")
    
    # Construir espaço fonético
    ps = PhoneticSpace(vocab)
    print(f"PhoneticSpace: {ps.vocab_size} fonemas × {ps.feature_dim} features\n")
    
    # Testar distâncias
    print("Distâncias:")
    pairs = [('e', 'ɛ'), ('o', 'ɔ'), ('p', 'b'), ('a', 'k'), ('a', 'e'), ('i', 'u')]
    for ph_a, ph_b in pairs:
        idx_a = vocab.p2i.get(ph_a, 1)
        idx_b = vocab.p2i.get(ph_b, 1)
        dist = ps.distance(idx_a, idx_b)
        error_class = ps.classify_error(idx_a, idx_b)
        print(f"  {ph_a} ↔ {ph_b}: {dist:.4f} (classe {error_class})")
    
    print("\nEscalas articulatórias:")
    for axis in ['altura', 'posicao', 'arredond', 'nasalidade', 'vozeamento']:
        scale = ps.articulatory_scale(axis)
        print(f"  {axis}:")
        for ph in ['a', 'e', 'i', 'o', 'u', 'ã']:
            idx = vocab.p2i.get(ph, 1)
            val = scale.get(idx, 0.5)
            print(f"    {ph}: {val:.2f}", end='  ')
        print()
    
    print("\nDistância no grafo (eixos altura + arredond):")
    graph_pairs = [('a', 'o'), ('a', 'u'), ('i', 'u')]
    for ph_a, ph_b in graph_pairs:
        idx_a = vocab.p2i.get(ph_a, 1)
        idx_b = vocab.p2i.get(ph_b, 1)
        dist = ps.graph_distance(idx_a, idx_b, axes=['altura', 'arredond'])
        print(f"  {ph_a} → {ph_b}: {dist:.4f}")
    
    print("\nResumo:")
    summary = ps.summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    print("\n✅ Teste concluído com sucesso!")
