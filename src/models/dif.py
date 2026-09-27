import math
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union
from joblib import Parallel, delayed


def _c_factor(n: int) -> float:
    """Calcula o fator de correção c(n) do Isolation Forest (Liu et al., 2012)."""
    if n > 2:
        return 2.0 * (math.log(n - 1) + 0.5772156649) - (2.0 * (n - 1) / n)
    elif n == 2:
        return 1.0
    else:
        return 0.0


class RandomProjectionNetwork(nn.Module):
    """Rede neural feedforward com pesos congelados para projeção aleatória não-linear (Phi_i)."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: Optional[int] = None,
        n_layers: int = 3,
        random_state: Optional[int] = None,
    ):
        super().__init__()
        if random_state is not None:
            torch.manual_seed(random_state)

        hidden_dim = hidden_dim or max(input_dim * 2, output_dim)
        layers: List[nn.Module] = []

        # Camada de entrada
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.LeakyReLU(negative_slope=0.2))

        # Camadas ocultas intermediárias (L - 2 camadas)
        for _ in range(max(0, n_layers - 2)):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LeakyReLU(negative_slope=0.2))

        # Camada de projeção final para R^d
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.network = nn.Sequential(*layers)

        # Inicialização Kaiming Normal e congelamento estrito dos pesos
        self._init_and_freeze_weights()
        self.eval()

    def _init_and_freeze_weights(self) -> None:
        for m in self.network.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0.2, mode="fan_in", nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        for p in self.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def project(self, X: np.ndarray, batch_size: int = 32768) -> np.ndarray:
        """Projeta matriz numpy X em R^d utilizando mini-batches."""
        self.eval()
        n_samples = X.shape[0]
        output_list: List[np.ndarray] = []

        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch = torch.from_numpy(X[start:end]).float()
            proj = self.network(batch).cpu().numpy()
            output_list.append(proj)

        return np.concatenate(output_list, axis=0) if output_list else np.empty((0, 0), dtype=np.float32)


class IsolationTreeNode:
    """Nó interno ou folha da árvore de isolamento."""

    def __init__(
        self,
        is_leaf: bool = False,
        size: int = 0,
        split_attr: int = -1,
        split_val: float = 0.0,
        min_val: float = 0.0,
        max_val: float = 0.0,
        left: Optional["IsolationTreeNode"] = None,
        right: Optional["IsolationTreeNode"] = None,
    ):
        self.is_leaf = is_leaf
        self.size = size
        self.split_attr = split_attr
        self.split_val = split_val
        self.min_val = min_val
        self.max_val = max_val
        self.left = left
        self.right = right


class IsolationTree:
    """Árvore de isolamento individual construída sobre o espaço de representação."""

    def __init__(self, max_depth: int = 8, random_state: Optional[int] = None):
        self.max_depth = max_depth
        self.rng = np.random.RandomState(random_state)
        self.root: Optional[IsolationTreeNode] = None

    def fit(self, X: np.ndarray, depth: int = 0) -> IsolationTreeNode:
        n_samples, n_features = X.shape

        if depth >= self.max_depth or n_samples <= 1:
            return IsolationTreeNode(is_leaf=True, size=n_samples)

        # Encontrar atributos que possuam variabilidade
        attrs = self.rng.permutation(n_features)
        split_attr = -1
        min_val, max_val = 0.0, 0.0

        for attr in attrs:
            attr_min = float(np.min(X[:, attr]))
            attr_max = float(np.max(X[:, attr]))
            if attr_max > attr_min:
                split_attr = attr
                min_val = attr_min
                max_val = attr_max
                break

        # Se todas as features são constantes nesta partição, encerra em folha
        if split_attr == -1:
            return IsolationTreeNode(is_leaf=True, size=n_samples)

        split_val = float(self.rng.uniform(min_val, max_val))
        left_mask = X[:, split_attr] < split_val
        right_mask = ~left_mask

        # Evitar partição degenerada se todos caírem no mesmo lado
        if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
            return IsolationTreeNode(is_leaf=True, size=n_samples)

        left_node = self.fit(X[left_mask], depth + 1)
        right_node = self.fit(X[right_mask], depth + 1)

        return IsolationTreeNode(
            is_leaf=False,
            size=n_samples,
            split_attr=split_attr,
            split_val=split_val,
            min_val=min_val,
            max_val=max_val,
            left=left_node,
            right=right_node,
        )

    def evaluate(self, X: np.ndarray, lambda_deas: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """Calcula h_deas e h_standard de forma vetorizada para um array X."""
        n_samples = X.shape[0]
        h_deas = np.zeros(n_samples, dtype=np.float64)
        h_standard = np.zeros(n_samples, dtype=np.float64)

        if self.root is None or n_samples == 0:
            return h_deas, h_standard

        # Rastreia as amostras que ainda estão percorrendo nós internos
        # Fila de processamento: (nó, índices das amostras que chegaram ao nó)
        stack: List[Tuple[IsolationTreeNode, np.ndarray]] = [(self.root, np.arange(n_samples))]

        eps = 1e-7

        while stack:
            node, indices = stack.pop()
            if len(indices) == 0:
                continue

            if node.is_leaf:
                cf = _c_factor(node.size)
                h_deas[indices] += cf
                h_standard[indices] += cf
            else:
                vals = X[indices, node.split_attr]
                # Cálculo do desvio relativo contínuo dev(x, T)
                # Intervalo do nó: [min_val, max_val]. Amostras de inferência usam os limites ampliados se fora da faixa
                eff_max = np.maximum(vals, node.max_val)
                eff_min = np.minimum(vals, node.min_val)
                range_span = eff_max - eff_min + eps
                dev = np.abs(vals - node.split_val) / range_span

                # Atualiza os comprimentos acumulados
                h_standard[indices] += 1.0
                h_deas[indices] += (1.0 - lambda_deas * dev)

                left_mask = vals < node.split_val
                right_mask = ~left_mask

                if np.any(left_mask) and node.left is not None:
                    stack.append((node.left, indices[left_mask]))
                if np.any(right_mask) and node.right is not None:
                    stack.append((node.right, indices[right_mask]))

        return h_deas, h_standard


class DeepIsolationForest:
    """Estimador Deep Isolation Forest (DIF) com suporte ao escore DEAS."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_samples: int = 256,
        n_representations: int = 10,
        n_layers: int = 3,
        projection_dim: Optional[int] = None,
        lambda_deas: float = 0.5,
        random_state: Optional[int] = 42,
        n_jobs: int = -1,
    ):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.n_representations = min(n_representations, n_estimators)
        self.n_layers = n_layers
        self.projection_dim = projection_dim
        self.lambda_deas = lambda_deas
        self.random_state = random_state
        self.n_jobs = n_jobs

        self.networks_: List[RandomProjectionNetwork] = []
        self.estimators_: List[Tuple[int, IsolationTree]] = []
        self.input_dim_: int = 0
        self.effective_proj_dim_: int = 0

    def fit(self, X: Union[np.ndarray, torch.Tensor]) -> "DeepIsolationForest":
        """Ajusta o ensemble exclusivamente sobre representações de tráfego legítimo normal."""
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
        X = np.asarray(X, dtype=np.float32)

        if X.ndim != 2:
            raise ValueError(f"X deve ser uma matriz 2D, recebido formato {X.shape}")

        n_samples, n_features = X.shape
        self.input_dim_ = n_features
        self.effective_proj_dim_ = self.projection_dim or n_features

        rng = np.random.RandomState(self.random_state)
        seeds = rng.randint(0, 1000000, size=self.n_estimators + self.n_representations)

        # 1. Instanciar as r redes de projeção aleatória não-linear
        self.networks_ = []
        for i in range(self.n_representations):
            net = RandomProjectionNetwork(
                input_dim=self.input_dim_,
                output_dim=self.effective_proj_dim_,
                n_layers=self.n_layers,
                random_state=int(seeds[i]),
            )
            self.networks_.append(net)

        # 2. Induzir as t árvores de isolamento
        max_depth = int(math.ceil(math.log2(max(self.max_samples, 2))))
        self.estimators_ = []

        subsample_size = min(self.max_samples, n_samples)

        for k in range(self.n_estimators):
            # Mapeia árvore k para uma rede no pool de representações
            net_idx = k % self.n_representations
            subsample_idx = rng.choice(n_samples, size=subsample_size, replace=False)
            X_sub = X[subsample_idx]

            # Projeta a subamostra no espaço z_i in R^d
            Z_sub = self.networks_[net_idx].project(X_sub)

            # Constrói a iTree
            tree = IsolationTree(max_depth=max_depth, random_state=int(seeds[self.n_representations + k]))
            tree.root = tree.fit(Z_sub)
            self.estimators_.append((net_idx, tree))

        return self

    def score_samples(
        self,
        X: Union[np.ndarray, torch.Tensor],
        lambda_deas: Optional[float] = None,
        batch_size: int = 32768,
    ) -> Dict[str, np.ndarray]:
        """Calcula os escores de anomalia normalizados em [0, 1] (DEAS e Standard)."""
        if not self.estimators_:
            raise RuntimeError("O estimador DeepIsolationForest deve ser ajustado com .fit() antes de prever.")

        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
        X = np.asarray(X, dtype=np.float32)

        lam = self.lambda_deas if lambda_deas is None else lambda_deas
        n_samples = X.shape[0]

        if n_samples == 0:
            return {
                "score_dif_deas": np.empty((0,), dtype=np.float64),
                "score_dif_standard": np.empty((0,), dtype=np.float64),
            }

        # 1. Pré-projetar o dataset através das r redes neurais (aproveitando GPU/CPU vetorizada)
        projected_reps: List[np.ndarray] = []
        for net in self.networks_:
            projected_reps.append(net.project(X, batch_size=batch_size))

        # 2. Avaliar as árvores em paralelo via joblib
        def _eval_tree(net_idx: int, tree: IsolationTree) -> Tuple[np.ndarray, np.ndarray]:
            Z = projected_reps[net_idx]
            return tree.evaluate(Z, lambda_deas=lam)

        results = Parallel(n_jobs=self.n_jobs, prefer="threads")(
            delayed(_eval_tree)(net_idx, tree) for net_idx, tree in self.estimators_
        )

        total_h_deas = np.zeros(n_samples, dtype=np.float64)
        total_h_standard = np.zeros(n_samples, dtype=np.float64)

        for h_d, h_s in results:
            total_h_deas += h_d
            total_h_standard += h_s

        mean_h_deas = total_h_deas / len(self.estimators_)
        mean_h_standard = total_h_standard / len(self.estimators_)

        c_psi = _c_factor(self.max_samples)
        if c_psi <= 0:
            c_psi = 1.0

        score_deas = np.power(2.0, -mean_h_deas / c_psi)
        score_standard = np.power(2.0, -mean_h_standard / c_psi)

        # Garantir limites [0, 1]
        score_deas = np.clip(score_deas, 0.0, 1.0)
        score_standard = np.clip(score_standard, 0.0, 1.0)

        return {
            "score_dif_deas": score_deas,
            "score_dif_standard": score_standard,
        }
