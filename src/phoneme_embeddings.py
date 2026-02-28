"""
Módulo de embeddings de fonemas para G2P.
Suporta múltiplos tipos: learned, panphon (features binárias).

Isolado de g2p.py para facilitar extensão e manutenção.
"""
import logging
import pickle
import subprocess
import sys
from pathlib import Path
import torch
import torch.nn as nn

logger = logging.getLogger("phoneme_embeddings")

# Símbolos IPA válidos, mas não-segmentais para o PanPhon FeatureTable.
# Ex.: acento primário/secundário, fronteira silábica, marcas prosódicas.
# Mantemos esses tokens no vocabulário (informação linguística), porém no
# embedding PanPhon fixo eles recebem vetor neutro (zeros).
NON_SEGMENTAL_IPA_SYMBOLS = {
    ".",   # fronteira silábica
    "ˈ",   # acento primário
    "ˌ",   # acento secundário
}

# Cache global para features do PanPhon (evita recarregar múltiplas vezes)
_PANPHON_CACHE_DIR = Path("cache")
_PANPHON_FT_CACHE = None


def _get_panphon_cache_path():
    """Retorna caminho do cache de features PanPhon."""
    return _PANPHON_CACHE_DIR / "panphon_feature_table.pkl"


def _load_panphon_with_subprocess():
    """
    Carrega PanPhon via subprocess com -X utf8.
    
    Estratégia: Executa Python com encoding UTF-8 forçado,
    carrega PanPhon, e retorna via pickle pela stdout.
    
    Isso isola o problema de encoding em um subprocess descartável.
    """
    script = """
import sys
import pickle
import panphon

ft = panphon.FeatureTable()

# Serializar e enviar via stdout
pickle.dump(ft, sys.stdout.buffer)
"""
    
    try:
        # Executar com -X utf8 (força UTF-8 mode)
        result = subprocess.run(
            [sys.executable, "-X", "utf8", "-c", script],
            capture_output=True,
            check=True,
            timeout=10
        )
        
        # Deserializar FeatureTable
        ft = pickle.loads(result.stdout)
        logger.info("PanPhon carregado via subprocess com UTF-8 mode")
        return ft
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Subprocess falhou: {e.stderr.decode('utf-8', errors='replace')}")
        raise RuntimeError(
            "Falha ao carregar PanPhon via subprocess.\n"
            f"Erro: {e.stderr.decode('utf-8', errors='replace')[:200]}"
        ) from e
    except Exception as e:
        logger.error(f"Erro inesperado no subprocess: {e}")
        raise


def _create_panphon_cache():
    """
    Cria cache persistente de FeatureTable do PanPhon.
    
    Cache é salvo em cache/panphon_feature_table.pkl
    Uma vez criado, nunca mais precisa de PanPhon!
    """
    cache_path = _get_panphon_cache_path()
    
    logger.info(f"Criando cache de PanPhon em {cache_path}...")
    
    # Carregar via subprocess
    ft = _load_panphon_with_subprocess()
    
    # Salvar cache
    cache_path.parent.mkdir(exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump(ft, f)
    
    logger.info(f"✅ Cache criado: {cache_path}")
    return ft


def _load_panphon_from_cache():
    """Carrega FeatureTable do cache (rápido, sem encoding issues)."""
    cache_path = _get_panphon_cache_path()
    
    if not cache_path.exists():
        return None
    
    try:
        with open(cache_path, 'rb') as f:
            ft = pickle.load(f)
        logger.debug(f"PanPhon carregado do cache: {cache_path}")
        return ft
    except Exception as e:
        logger.warning(f"Falha ao carregar cache: {e}")
        return None


def _load_panphon_with_fallback():
    """
    Carrega PanPhon FeatureTable com estratégia inteligente em camadas.
    
    Estratégia em ordem (do mais eficiente para menos):
    1. Cache persistente (se existe)
    2. Import normal (funciona em alguns ambientes)
    3. Subprocess com -X utf8 + criar cache
    4. Erro descritivo
    
    Returns:
        panphon.FeatureTable instance
    
    Raises:
        RuntimeError: Se não conseguir carregar de forma alguma
    """
    global _PANPHON_FT_CACHE
    
    # Se já carregou nesta sessão, reusar
    if _PANPHON_FT_CACHE is not None:
        return _PANPHON_FT_CACHE
    
    # Estratégia 1: Carregar do cache (mais rápido)
    ft = _load_panphon_from_cache()
    if ft is not None:
        _PANPHON_FT_CACHE = ft
        return ft
    
    # Estratégia 2: Import normal (pode funcionar em Linux/Mac)
    try:
        import panphon
        ft = panphon.FeatureTable()
        logger.debug("PanPhon carregado normalmente")
        _PANPHON_FT_CACHE = ft
        
        # Criar cache para próxima vez (otimização)
        try:
            _create_panphon_cache()
        except:
            pass  # Não é crítico
        
        return ft
    except UnicodeDecodeError:
        logger.debug("Import normal falhou (encoding Windows), tentando subprocess...")
    except ImportError as e:
        raise ImportError(
            "PanPhon não instalado. Instale com: pip install panphon"
        ) from e
    
    # Estratégia 3: Subprocess + cache
    try:
        logger.info("Carregando PanPhon via subprocess (primeira vez, pode demorar ~5s)...")
        ft = _create_panphon_cache()
        _PANPHON_FT_CACHE = ft
        return ft
    except Exception as e:
        logger.error(f"Todas estratégias falharam: {e}")
        
        # Estratégia 4: Erro descritivo
        raise RuntimeError(
            "Não foi possível carregar PanPhon FeatureTable.\n"
            f"Erro: {e}\n\n"
            "Soluções possíveis:\n"
            "1. MAIS SIMPLES: Use embedding_type='learned' no config\n"
            "2. Ou crie cache manualmente:\n"
            "   python -X utf8 -c \"from src.phoneme_embeddings import _create_panphon_cache; _create_panphon_cache()\"\n"
            "3. Ou instale pandas/panphon compatíveis\n"
        ) from e


class LearnedPhonemeEmbedding(nn.Module):
    """
    Embedding tradicional aprendido (backward compatible).
    
    Wrapper do nn.Embedding padrão do PyTorch. Usado nos experimentos
    anteriores (exp1, exp2, exp3).
    """
    def __init__(self, vocab_size, emb_dim, padding_idx=0):
        """
        Args:
            vocab_size: Tamanho do vocabulário de fonemas
            emb_dim: Dimensão do embedding
            padding_idx: Índice do token PAD (default: 0)
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=padding_idx)
        self.embedding_dim = emb_dim  # Expor embedding_dim como atributo (padrão nn.Embedding)
        self.emb_dim = emb_dim
        self.vocab_size = vocab_size
        self.padding_idx = padding_idx
    
    def forward(self, x):
        """
        Args:
            x: Tensor de índices (batch, seq_len)
        
        Returns:
            Tensor de embeddings (batch, seq_len, emb_dim)
        """
        return self.embedding(x)


class PanPhonPhonemeEmbedding(nn.Module):
    """
    Embedding baseado em features fonéticas binárias do PanPhon.
    
    Em vez de vetores aprendidos, usa as 24 features articulatórias
    binárias do PanPhon (syl, son, cons, cont, delrel, lat, nas, strid,
    voi, sg, cg, ant, cor, distr, lab, hi, lo, back, round, velaric,
    tense, long, hitone, hireg).
    
    Features são fixas (não aprendidas), fornecendo conhecimento
    fonético a priori ao modelo.
    """
    def __init__(self, vocab_size, i2p_mapping, padding_idx=0):
        """
        Args:
            vocab_size: Tamanho do vocabulário de fonemas
            i2p_mapping: Dict {idx: phoneme_str} para mapear índices → fonemas
            padding_idx: Índice do token PAD (default: 0)
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.padding_idx = padding_idx
        
        # Carregar PanPhon com fallback automático
        self.ft = _load_panphon_with_fallback()
        
        # Construir matriz de features (vocab_size, 24)
        # register_buffer = não é parâmetro treinável, mas salvo no state_dict
        feature_matrix = self._build_feature_matrix(i2p_mapping)
        self.register_buffer('feature_matrix', feature_matrix)
        self.emb_dim = self.feature_matrix.shape[1]  # Deve ser 24
        self.embedding_dim = self.emb_dim  # Alias para consistência com nn.Embedding
        
        logger.info(
            f"PanPhonPhonemeEmbedding: vocab_size={vocab_size}, "
            f"emb_dim={self.emb_dim} (features fonéticas)"
        )
    
    def _build_feature_matrix(self, i2p_mapping):
        """
        Constrói matriz [vocab_size, 24] de features binárias.
        
        Args:
            i2p_mapping: Dict {idx: phoneme_str}
        
        Returns:
            Tensor (vocab_size, 24) com features binárias (0/-1 → 0/1)
        """
        n_features = 24  # PanPhon tem 24 features
        matrix = torch.zeros(self.vocab_size, n_features, dtype=torch.float32)
        
        unk_count = 0
        non_segmental_count = 0
        special_tokens = {'<PAD>', '<UNK>', '<EOS>'}
        
        for idx, phoneme in i2p_mapping.items():
            if idx == self.padding_idx:
                # PAD = vetor zero (não contribui para atenção)
                continue
            
            if phoneme in special_tokens:
                # Tokens especiais: vetor zero por enquanto
                # Alternativa futura: vetor médio de todos os fonemas
                continue

            if phoneme in NON_SEGMENTAL_IPA_SYMBOLS:
                # Símbolos prosódicos/estruturais válidos no IPA.
                # PanPhon modela segmentos; portanto usamos vetor neutro.
                non_segmental_count += 1
                continue
            
            # Extrair features do PanPhon
            try:
                # word_to_vector_list retorna lista de vetores (um por segmento)
                # Para fonema único: pegar primeiro vetor
                # Valores originais: 0 (não especificado), -1 (negativo), +1 (positivo)
                vec = self.ft.word_to_vector_list(phoneme, numeric=True)
                
                if vec and len(vec) > 0:
                    # Normalizar: transforma [-1, 0, 1] → [0, 0.5, 1]
                    # Preserva informação ternária (- / 0 / +)
                    features = torch.tensor(vec[0], dtype=torch.float32)
                    # Mapear: -1→0, 0→0.5, 1→1
                    features = (features + 1) / 2.0
                    matrix[idx] = features
                else:
                    unk_count += 1
                    logger.warning(
                        f"PanPhon retornou vetor vazio para '{phoneme}' (idx={idx})"
                    )
            
            except Exception as e:
                # Fallback se PanPhon não reconhecer o fonema
                unk_count += 1
                logger.warning(
                    f"PanPhon não reconheceu '{phoneme}' (idx={idx}): {e}"
                )
                # Deixa como vetor zero (pode ser refinado com média mais tarde)
        
        if unk_count > 0:
            logger.warning(
                f"PanPhon: {unk_count} fonemas não reconhecidos "
                f"(de {self.vocab_size - len(special_tokens)} não-especiais)"
            )

        if non_segmental_count > 0:
            logger.info(
                "PanPhon: %d símbolo(s) IPA não-segmentais tratados com vetor neutro",
                non_segmental_count,
            )
        
        return matrix
    
    def forward(self, x):
        """
        Args:
            x: Tensor de índices (batch, seq_len)
        
        Returns:
            Tensor de features (batch, seq_len, 24)
        """
        # Lookup direto na matriz (sem gradiente, features fixas)
        return self.feature_matrix[x]


class PanPhonTrainableEmbedding(nn.Module):
    """
    Embedding PanPhon com projeção trainável 24D → emb_dim.
    
    Usa as 24 features articulatórias fixas do PanPhon como base,
    e projeta para uma dimensão maior via nn.Linear trainável.
    
    Isso combina o inductive bias linguístico com a capacidade de
    aprender representações mais ricas durante o treinamento.
    
    A matriz de features base (24D) é fixa (register_buffer),
    mas a projeção linear (24 → emb_dim) é treinável.
    """
    def __init__(self, vocab_size, emb_dim, i2p_mapping, padding_idx=0):
        """
        Args:
            vocab_size: Tamanho do vocabulário de fonemas
            emb_dim: Dimensão de saída do embedding (ex: 128)
            i2p_mapping: Dict {idx: phoneme_str} para mapear índices → fonemas
            padding_idx: Índice do token PAD (default: 0)
        """
        super().__init__()
        
        # Criar a base fixa de PanPhon (24D)
        self._panphon_base = PanPhonPhonemeEmbedding(
            vocab_size, i2p_mapping, padding_idx
        )
        
        # Projeção trainável: 24D → emb_dim
        self.projection = nn.Linear(24, emb_dim)
        
        # Expor dimensões para compat com Decoder
        self.emb_dim = emb_dim
        self.embedding_dim = emb_dim
        self.vocab_size = vocab_size
        self.padding_idx = padding_idx
        
        logger.info(
            f"PanPhonTrainableEmbedding: vocab_size={vocab_size}, "
            f"24D PanPhon → {emb_dim}D (projeção trainável)"
        )
    
    def forward(self, x):
        """
        Args:
            x: Tensor de índices (batch, seq_len)
        
        Returns:
            Tensor de embeddings (batch, seq_len, emb_dim)
        """
        # Features fixas do PanPhon (24D, sem gradiente)
        base_features = self._panphon_base(x)  # (batch, seq_len, 24)
        # Projeção trainável para dimensão alvo
        return self.projection(base_features)   # (batch, seq_len, emb_dim)


def get_embedding_layer(embedding_type, vocab_size, emb_dim=None, 
                        i2p_mapping=None, padding_idx=0):
    """
    Factory function para criar camada de embedding de fonemas.
    
    Centraliza a lógica de criação para facilitar extensão futura
    (ex: hybrid embeddings, transformer embeddings, etc.).
    
    Args:
        embedding_type: Tipo de embedding
            - "learned": Embedding tradicional aprendido (nn.Embedding)
            - "panphon": Features binárias do PanPhon
                - Se emb_dim omitido ou == 24: 24D fixo (non-trainable)
                - Se emb_dim > 24: 24D PanPhon → emb_dim via projeção trainável
        vocab_size: Tamanho do vocabulário de fonemas
        emb_dim: Dimensão do embedding (obrigatório para "learned",
                 para "panphon": >24 ativa projeção trainável)
        i2p_mapping: Dict {idx: phoneme_str} (obrigatório para "panphon")
        padding_idx: Índice do token PAD (default: 0)
    
    Returns:
        nn.Module com método forward(x) → (batch, seq_len, emb_dim)
    
    Raises:
        ValueError: Se argumentos obrigatórios estão faltando ou tipo desconhecido
    
    Examples:
        >>> # Embedding aprendido (exp0, exp1, exp2)
        >>> emb = get_embedding_layer("learned", vocab_size=50, emb_dim=128)
        >>> emb.emb_dim
        128
        
        >>> # Embedding PanPhon trainável (exp3)
        >>> i2p = {0: '<PAD>', 1: '<UNK>', 2: '<EOS>', 3: 'a', 4: 'e'}
        >>> emb = get_embedding_layer("panphon", vocab_size=50, emb_dim=128, i2p_mapping=i2p)
        >>> emb.emb_dim  # 24 → 128 via projeção trainável
        128
        
        >>> # Embedding PanPhon fixo (exp4)
        >>> emb = get_embedding_layer("panphon", vocab_size=50, i2p_mapping=i2p)
        >>> emb.emb_dim
        24
    """
    if embedding_type == "learned":
        if emb_dim is None:
            raise ValueError(
                "emb_dim é obrigatório para embedding_type='learned'. "
                "Especifique a dimensão do embedding (ex: 128, 256)."
            )
        return LearnedPhonemeEmbedding(vocab_size, emb_dim, padding_idx)
    
    elif embedding_type == "panphon":
        if i2p_mapping is None:
            raise ValueError(
                "i2p_mapping é obrigatório para embedding_type='panphon'. "
                "Passe o dicionário {idx: phoneme_str} do vocabulário."
            )
        # Se emb_dim especificado e > 24: modo trainável com projeção
        # Se emb_dim omitido ou == 24: modo fixo original (24D puro)
        if emb_dim is not None and emb_dim > 24:
            return PanPhonTrainableEmbedding(
                vocab_size, emb_dim, i2p_mapping, padding_idx
            )
        return PanPhonPhonemeEmbedding(vocab_size, i2p_mapping, padding_idx)
    
    else:
        raise ValueError(
            f"embedding_type desconhecido: '{embedding_type}'. "
            f"Use 'learned' ou 'panphon'."
        )
