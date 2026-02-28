"""
Phonetic Distance-Aware Loss Functions for G2P

Implements loss functions that penalize phoneme errors based on phonetic distance,
inspired by structured prediction and metric learning.

Referências:
- Tsochantaridis et al. (2005): Structured Prediction with SVMs
- Schroff et al. (2015): FaceNet - Metric Learning with Triplet Loss
- Szegedy et al. (2016): Label Smoothing / Knowledge Distillation
- Bahdanau et al. (2014): Attention Mechanisms
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple


class PhonicDistanceAwareLoss(nn.Module):
    """
    Loss function that penalizes phonetic errors proportionally to their
    phonetic distance in articulatory feature space (PanPhon).
    
    Theory:
    -------
    Standard Cross-Entropy loss treats all errors equally:
        L_CE = -log(p_correct)
        
    This is unfair: predicting /ɛ/ instead of /e/ (1 feature difference, ~0.2 distance)
    should cost less than predicting /k/ instead of /e/ (8+ feature difference, ~0.8 distance).
    
    Solution: Add distance-weighted penalty term
        L_DAPL = L_CE + λ · Σ(d_panphon(pred_i, ref_i) · p(pred_i))
        
    Where:
        - d_panphon: Euclidean distance in 24D articulatory feature space
        - p(pred_i): Probability the model assigned to incorrect prediction
        - λ: Weight balancing CE vs distance term (hyperparameter)
    
    Effect:
        - Errors on phonetically close phonemes: smaller penalty
        - Errors on phonetically distant phonemes: larger penalty
        - Naturally implements soft targets (like knowledge distillation)
    
    Benefit for training:
        - Faster convergence (softer gradient landscape)
        - Better generalization (respects phonetic structure)
        - Linguistically motivated (matches how linguists think of errors)
    
    Attributes
    ----------
    distance_matrix : torch.Tensor
        Pre-computed [n_phonemes, n_phonemes] matrix of pairwise distances
    distance_lambda : float
        Weight of distance term (0.0 = standard CE, >0.5 = strong constraints)
    distance_metric : str
        Metric for computing distances {'euclidean', 'cosine', 'manhattan'}
    device : torch.device
        GPU/CPU device for tensors
    """
    
    def __init__(self,
                 phoneme_vocab: Dict[str, int],
                 panphon_features: Dict[str, np.ndarray],
                 distance_lambda: float = 0.1,
                 distance_metric: str = 'euclidean',
                 normalize_distance: bool = True):
        """
        Initialize Phonetic Distance-Aware Loss.
        
        Parameters
        ----------
        phoneme_vocab : Dict[str, int]
            Phoneme to index mapping. Example:
            {'/p/': 0, '/b/': 1, '/a/': 2, ...}
        
        panphon_features : Dict[str, np.ndarray]
            Phoneme to 24D feature vector. Example:
            {'/p/': array([0, -1, -1, ...]), '/a/': array([1, 0, -1, ...]), ...}
        
        distance_lambda : float
            Regularization weight for distance term.
            - 0.0: Pure CE loss (baseline)
            - 0.1: Weak constraint (default, 10% distance penalty)
            - 0.5: Medium constraint (50-50 CE vs distance)
            - 1.0: Strong constraint (structure dominates)
            
            Recommended: 0.01-0.5 (empirically determined via ablation)
        
        distance_metric : str
            How to compute distances between phonemes:
            - 'euclidean': √(Σ(f_i - f_i')²) — standard, interpretable
            - 'cosine': 1 - (f·f')/(|f||f'|) — normalized, better for high dims
            - 'manhattan': Σ|f_i - f_i'| — L1, more robust to outliers
            
            Recommended: 'euclidean' (phonetically meaningful)
        
        normalize_distance : bool
            If True, divide all distances by max_distance to get [0, 1] scale.
            Advantages: learnable λ becomes scale-invariant, interpretable.
        """
        super().__init__()
        
        self.distance_lambda = distance_lambda
        self.distance_metric = distance_metric
        self.normalize_distance = normalize_distance
        
        # Build distance matrix: [n_phonemes, n_phonemes]
        self.distance_matrix = self._build_distance_matrix(
            phoneme_vocab, panphon_features, distance_metric
        )
        
        # Normalize to [0, 1]
        if normalize_distance:
            max_dist = self.distance_matrix.max()
            if max_dist > 0:
                self.distance_matrix = self.distance_matrix / max_dist

        # --- Post-hoc: distâncias customizadas para símbolos estruturais ---
        # DEVE ser aplicado APÓS a normalização: o override de 1.0 representa
        # distância máxima na escala [0,1] normalizada. Se aplicado antes,
        # a divisão por max_dist euclidiano (~3-5) reduziria o valor para ~0.2-0.3,
        # tornando-o equivalente a fonemas similares (ex: ɛ↔e ≈ 0.25).
        structural_symbols = {'.', 'ˈ'}
        for sym, idx_sym in phoneme_vocab.items():
            if sym not in structural_symbols:
                continue
            for other, idx_other in phoneme_vocab.items():
                if other == sym:
                    continue  # identidade: 0.0 (não tocar)
                self.distance_matrix[idx_sym, idx_other] = 1.0
                self.distance_matrix[idx_other, idx_sym] = 1.0
        # --- fim overrides estruturais ---

        # Register as buffer (moves to GPU automatically with model)
        self.register_buffer('_distance_matrix', self.distance_matrix)
        
        # Standard CE loss (PyTorch default)
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=0, reduction='none')
        
        # Shape info for debugging
        self.vocab_size = len(phoneme_vocab)
    
    def _build_distance_matrix(self,
                               phoneme_vocab: Dict[str, int],
                               panphon_features: Dict[str, np.ndarray],
                               metric: str = 'euclidean') -> torch.Tensor:
        """
        Pre-compute pairwise phonetic distances between all phonemes.
        
        Example for PT-BR (43 phonemes):
        dist[3, 8] = distance(/p/, /b/) ≈ 2.0  (differ only in voicing)
        dist[3, 15] = distance(/p/, /k/) ≈ 3.5 (differ in place + manner)
        
        Parameters
        ----------
        phoneme_vocab : Dict[str, int]
            Phoneme to vocab index
        
        panphon_features : Dict[str, np.ndarray]
            Phoneme to feature vector (24D articulatory)
        
        metric : str
            Distance metric to use
        
        Returns
        -------
        torch.Tensor
            [n_phonemes, n_phonemes] distance matrix
        """
        n_phonemes = len(phoneme_vocab)
        distance_matrix = torch.zeros(n_phonemes, n_phonemes)
        
        # Iterate over all phoneme pairs
        for phon_i, idx_i in phoneme_vocab.items():
            if phon_i not in panphon_features:
                # Fallback for OOV phonemes: use zero distance (avoid crashes)
                # Could also use centroid or nearest neighbor
                continue
            
            feat_i = panphon_features[phon_i]  # 24D array
            
            for phon_j, idx_j in phoneme_vocab.items():
                if phon_j not in panphon_features:
                    continue
                
                feat_j = panphon_features[phon_j]  # 24D array
                
                # Compute distance based on metric
                if metric == 'euclidean':
                    dist = float(np.sqrt(np.sum((feat_i - feat_j) ** 2)))
                
                elif metric == 'cosine':
                    # Cosine distance (1 - similarity)
                    dot_product = np.dot(feat_i, feat_j)
                    norm_i = np.linalg.norm(feat_i)
                    norm_j = np.linalg.norm(feat_j)
                    
                    if norm_i > 0 and norm_j > 0:
                        cosine_sim = dot_product / (norm_i * norm_j)
                        dist = float(1.0 - cosine_sim)
                    else:
                        dist = 0.0
                
                elif metric == 'manhattan':
                    # L1 distance
                    dist = float(np.sum(np.abs(feat_i - feat_j)))
                
                else:
                    raise ValueError(f"Unknown metric: {metric}")
                
                distance_matrix[idx_i, idx_j] = dist

        return distance_matrix
    
    def forward(self,
                logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        """
        Compute phonetic distance-aware loss.
        
        Parameters
        ----------
        logits : torch.Tensor
            Model output logits, shape [batch_size, seq_len, vocab_size]
            Typically unnormalized scores from output layer
        
        targets : torch.Tensor
            Gold phoneme labels, shape [batch_size, seq_len]
            Values in [0, vocab_size), where 0 = padding token (ignored)
        
        Returns
        -------
        torch.Tensor
            Scalar loss value (averaged over batch)
        
        Computation
        -----------
        For each timestep where pred ≠ target:
            loss_t = L_CE(logits_t, target_t) + λ · d_panphon(pred_t, target_t) · p(pred_t)
        
        Where:
            - L_CE = -log(p(target))
            - d_panphon = normalized Euclidean distance in feature space
            - p(pred) = softmax probability of predicted phoneme
            - λ = distance_lambda (weight)
        """
        batch_size, seq_len, vocab_size = logits.shape
        
        # Flatten for easier computation: [batch*seq, vocab]
        logits_2d = logits.view(-1, vocab_size)
        targets_1d = targets.view(-1)
        
        # ============ Cross-Entropy Loss ============
        # Shape: [batch*seq]
        ce_loss_per_token = self.ce_loss(logits_2d, targets_1d)
        
        # ============ Phonetic Distance Loss ============
        # Get predicted phoneme indices from logits
        pred_phonemes = logits_2d.argmax(dim=1)  # [batch*seq]
        
        # Get normalized probabilities
        probs = F.softmax(logits_2d, dim=1)  # [batch*seq, vocab]
        
        # Get probability of each prediction: p(ŷ_t)
        # Shape: [batch*seq]
        # gather() é idiomático PyTorch: sem alocação de torch.arange() a cada batch (2026-02-23)
        pred_probs = probs.gather(1, pred_phonemes.unsqueeze(1)).squeeze(1)
        
        # Get phonetic distances: d(pred_t, target_t)
        # Using registered buffer (automatically on correct device)
        distances = self._distance_matrix[pred_phonemes, targets_1d]  # [batch*seq]
        
        # Distance penalty = distance × probability
        # Intuition: high probability + wrong = larger penalty
        #           low probability + wrong = smaller penalty (model wasn't confident)
        distance_loss_per_token = distances * pred_probs
        
        # ============ Combine Losses ============
        # Mask out padding tokens (target = 0)
        mask = (targets_1d != 0).float()  # [batch*seq], 1 for non-padding
        
        # Combined loss
        combined_loss = ce_loss_per_token + self.distance_lambda * distance_loss_per_token
        
        # Apply mask and average
        masked_loss = (combined_loss * mask).sum()
        n_valid_tokens = mask.sum()
        
        # Avoid division by zero
        final_loss = masked_loss / (n_valid_tokens + 1e-8)
        
        return final_loss
    
    def forward_debug(self,
                      logits: torch.Tensor,
                      targets: torch.Tensor,
                      phoneme_idx2str: Optional[Dict[int, str]] = None) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass with debugging information.
        
        Useful for:
        - Understanding which phonemes contribute most to loss
        - Monitoring distance distribution during training
        - Detecting mode collapse (model predicts same phoneme)
        
        Returns
        -------
        loss : torch.Tensor
            Scalar loss (same as forward())
        
        debug_info : Dict
            Contains:
            - 'ce_loss': Mean CE component
            - 'distance_loss': Mean distance component
            - 'mean_distance': Average distance between pred & target
            - 'top_errors': Top 10 hardest predictions
            - 'distance_stats': Min, max, mean, std of distances
        """
        batch_size, seq_len, vocab_size = logits.shape
        logits_2d = logits.view(-1, vocab_size)
        targets_1d = targets.view(-1)
        
        # Standard forward
        loss = self.forward(logits, targets)
        
        # Debug info
        ce_loss_per_token = self.ce_loss(logits_2d, targets_1d)
        pred_phonemes = logits_2d.argmax(dim=1)
        probs = F.softmax(logits_2d, dim=1)
        pred_probs = probs.gather(1, pred_phonemes.unsqueeze(1)).squeeze(1)
        distances = self._distance_matrix[pred_phonemes, targets_1d]

        mask = (targets_1d != 0).float()

        debug_info = {
            'ce_loss': (ce_loss_per_token * mask).mean().item(),
            'distance_loss': (distances * pred_probs * mask).mean().item(),
            'mean_distance': (distances * mask).mean().item(),
            'distance_stats': {
                'min': distances.min().item(),
                'max': distances.max().item(),
                'mean': distances.mean().item(),
                'std': distances.std().item(),
            }
        }
        
        # Top 10 hardest (highest combined loss)
        combined = ce_loss_per_token + self.distance_lambda * distances * pred_probs
        top_k = torch.topk(combined, k=min(10, len(combined)))
        
        if phoneme_idx2str:
            debug_info['top_errors'] = [
                (phoneme_idx2str.get(pred_phonemes[i].item(), '?'),
                 phoneme_idx2str.get(targets_1d[i].item(), '?'),
                 combined[i].item())
                for i in top_k.indices.tolist()
            ]
        
        return loss, debug_info


class SoftTargetCrossEntropyLoss(nn.Module):
    """
    Alternative approach: Convert hard targets to soft targets based on phonetic distance.
    
    Instead of one-hot targets [0, 0, 1, 0], create soft distribution:
    [0.02, 0.08, 0.80, 0.10]  (neighboring phonemes get small positive weights)
    
    Theory
    ------
    This is essentially label smoothing with phonetic distance as kernel.
    
    Reference:
    - Szegedy et al. (2016): "Rethinking Inception Architecture" (label smoothing)
    - Zhou et al. (2017): "Surface Structures" (related to soft targets)
    
    Advantages:
    - Mathematically cleaner (standard KL divergence on probability distributions)
    - More amenable to theoretical analysis
    
    Disadvantages:
    - Slightly higher computational cost (need to create soft targets for each batch)
    - May over-smooth if temperature too high
    """
    
    def __init__(self,
                 phoneme_vocab: Dict[str, int],
                 panphon_features: Dict[str, np.ndarray],
                 temperature: float = 0.5,
                 distance_metric: str = 'euclidean'):
        """
        Parameters
        ----------
        temperature : float
            Controls how soft the targets are.
            - Low (0.1): Strong peaking on correct phoneme
            - Medium (0.5): Some neighboring support
            - High (2.0): Very smooth distribution
        """
        super().__init__()
        
        self.temperature = temperature
        
        # Build normalized distance matrix
        distance_matrix = self._build_distance_matrix(
            phoneme_vocab, panphon_features, distance_metric
        )
        self.register_buffer('_distance_matrix', distance_matrix)
        
        self.vocab_size = len(phoneme_vocab)
    
    def _build_distance_matrix(self,
                               phoneme_vocab: Dict[str, int],
                               panphon_features: Dict[str, np.ndarray],
                               metric: str = 'euclidean') -> torch.Tensor:
        """Same as PhonicDistanceAwareLoss"""
        n_phonemes = len(phoneme_vocab)
        distance_matrix = torch.zeros(n_phonemes, n_phonemes)
        
        for phon_i, idx_i in phoneme_vocab.items():
            if phon_i not in panphon_features:
                continue
            feat_i = panphon_features[phon_i]
            
            for phon_j, idx_j in phoneme_vocab.items():
                if phon_j not in panphon_features:
                    continue
                feat_j = panphon_features[phon_j]
                
                if metric == 'euclidean':
                    dist = float(np.sqrt(np.sum((feat_i - feat_j) ** 2)))
                else:
                    raise NotImplementedError
                
                distance_matrix[idx_i, idx_j] = dist
        
        # Normalize
        max_dist = distance_matrix.max()
        if max_dist > 0:
            distance_matrix = distance_matrix / max_dist
        
        return distance_matrix
    
    def forward(self,
                logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        """
        Create soft targets and compute KL divergence.
        
        Soft target creation:
            soft_targets[i] = softmax(-distances[i] / temperature)
        
        This makes neighboring phonemes have reasonable probability.
        """
        batch_size, seq_len, vocab_size = logits.shape
        
        logits_2d = logits.view(-1, vocab_size)
        targets_1d = targets.view(-1)
        
        # Create soft targets: [batch*seq, vocab]
        distances = self._distance_matrix[targets_1d]  # [batch*seq, vocab]
        soft_targets = F.softmax(-distances / self.temperature, dim=1)
        
        # Log probabilities from model
        log_probs = F.log_softmax(logits_2d, dim=1)
        
        # KL divergence: sum over vocab
        kl_loss = -(soft_targets * log_probs).sum(dim=1)  # [batch*seq]
        
        # Mask padding and average
        mask = (targets_1d != 0).float()
        loss = (kl_loss * mask).sum() / (mask.sum() + 1e-8)
        
        return loss


class SequenceCrossEntropyLoss(nn.Module):
    """Wrapper around nn.CrossEntropyLoss that accepts 3D sequence logits.
    
    Unifies the interface so all loss functions accept the same shapes:
        logits:  (batch, seq_len, vocab_size)  or  (N, vocab_size)
        targets: (batch, seq_len)              or  (N,)
    
    This eliminates the need for isinstance checks in the training loop,
    since nn.CrossEntropyLoss requires pre-flattened (N, C) input while
    custom losses (PhonicDistanceAwareLoss, SoftTargetCrossEntropyLoss)
    handle flattening internally.
    """
    def __init__(self, ignore_index: int = 0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.dim() == 3:
            logits = logits.reshape(-1, logits.size(-1))
            targets = targets.reshape(-1)
        return self.ce(logits, targets)


# Factory function for clean integration in train.py
def get_loss_function(loss_type: str,
                      phoneme_vocab: Optional[Dict[str, int]] = None,
                      panphon_features: Optional[Dict[str, np.ndarray]] = None,
                      config: Optional[Dict] = None) -> nn.Module:
    """
    Factory function to instantiate appropriate loss function.
    
    All returned losses accept the same interface:
        loss(logits, targets) where logits is (batch, seq_len, vocab_size).
    
    Parameters
    ----------
    loss_type : str
        Type of loss: 'cross_entropy', 'distance_aware', 'soft_target'
    
    phoneme_vocab : Optional[Dict[str, int]]
        Phoneme vocabulary (required for 'distance_aware' and 'soft_target')
    
    panphon_features : Optional[Dict[str, np.ndarray]]
        PanPhon features for each phoneme (required for 'distance_aware' and 'soft_target')
    
    config : Optional[Dict]
        Hyperparameters for the loss (e.g., {'distance_lambda': 0.1})
    
    Returns
    -------
    nn.Module
        Instantiated loss function with unified (batch, seq, vocab) interface
    
    Example
    -------
    >>> criterion = get_loss_function('cross_entropy')
    >>> criterion = get_loss_function(
    ...     'distance_aware',
    ...     phoneme_vocab={'a': 0, 'e': 1, ...},
    ...     panphon_features={'a': array([...]), 'e': array([...]), ...},
    ...     config={'distance_lambda': 0.1}
    ... )
    """
    if config is None:
        config = {}
    
    if loss_type == 'cross_entropy':
        return SequenceCrossEntropyLoss(ignore_index=0)
    
    elif loss_type == 'distance_aware':
        distance_lambda = config.get('distance_lambda', 0.1)
        distance_metric = config.get('distance_metric', 'euclidean')
        
        return PhonicDistanceAwareLoss(
            phoneme_vocab=phoneme_vocab,
            panphon_features=panphon_features,
            distance_lambda=distance_lambda,
            distance_metric=distance_metric,
            normalize_distance=True
        )
    
    elif loss_type == 'soft_target':
        temperature = config.get('temperature', 0.5)
        distance_metric = config.get('distance_metric', 'euclidean')
        
        return SoftTargetCrossEntropyLoss(
            phoneme_vocab=phoneme_vocab,
            panphon_features=panphon_features,
            temperature=temperature,
            distance_metric=distance_metric
        )
    
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
