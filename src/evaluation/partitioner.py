import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from sklearn.model_selection import train_test_split


class DataPartitioner:
    """Particionador estratificado livre de vazamento de dados (anti-leakage guard)."""

    def __init__(
        self,
        train_normal_ratio: float = 0.70,
        val_normal_ratio: float = 0.15,
        test_normal_ratio: float = 0.15,
        val_attack_ratio: float = 0.20,
        rare_threshold: int = 15,
        random_state: Optional[int] = 42,
    ):
        if not np.isclose(train_normal_ratio + val_normal_ratio + test_normal_ratio, 1.0):
            raise ValueError("As proporções das amostras normais devem somar 1.0.")

        self.train_normal_ratio = train_normal_ratio
        self.val_normal_ratio = val_normal_ratio
        self.test_normal_ratio = test_normal_ratio
        self.val_attack_ratio = val_attack_ratio
        self.test_attack_ratio = 1.0 - val_attack_ratio
        self.rare_threshold = rare_threshold
        self.random_state = random_state

    def split(
        self,
        df: pd.DataFrame,
        label_column: str = "Label",
        normal_label: Any = "0",
    ) -> Dict[str, Any]:
        """Divide o dataset preservando tráfego normal em 70/15/15 e ataques estratificados em 20/80."""
        rng = np.random.RandomState(self.random_state)

        # 1. Identificar tráfego normal vs tráfego de ataque
        labels_str = df[label_column].astype(str).str.strip()
        normal_str = str(normal_label).strip()

        is_normal = (
            (labels_str == normal_str) |
            (labels_str.str.upper() == "BENIGN") |
            (labels_str.str.upper() == "NORMAL")
        )

        df_normal = df[is_normal].copy()
        df_attack = df[~is_normal].copy()

        # 2. Particionar amostras normais: 70% treino, 15% validação, 15% teste
        n_normal = len(df_normal)
        shuffled_normal_idx = rng.permutation(df_normal.index.to_numpy())

        n_train_norm = int(round(self.train_normal_ratio * n_normal))
        n_val_norm = int(round(self.val_normal_ratio * n_normal))

        idx_train_norm = shuffled_normal_idx[:n_train_norm]
        idx_val_norm = shuffled_normal_idx[n_train_norm : n_train_norm + n_val_norm]
        idx_test_norm = shuffled_normal_idx[n_train_norm + n_val_norm :]

        df_train_norm = df.loc[idx_train_norm]
        df_val_norm = df.loc[idx_val_norm]
        df_test_norm = df.loc[idx_test_norm]

        # 3. Particionar ataques de forma estratificada por tipo de classe: 20% val / 80% teste
        df_val_attacks_list = []
        df_test_attacks_list = []
        rare_classes: Dict[str, int] = {}

        if len(df_attack) > 0:
            attack_groups = df_attack.groupby(label_column, observed=True)

            for class_name, group in attack_groups:
                count = len(group)
                if count < self.rare_threshold:
                    rare_classes[str(class_name)] = count

                grp_indices = rng.permutation(group.index.to_numpy())


                # Garantir pelo menos 1 amostra de validação se houver mais de 1 amostra
                n_val_att = int(round(self.val_attack_ratio * count))
                if count > 1 and n_val_att == 0:
                    n_val_att = 1
                if count > 1 and n_val_att == count:
                    n_val_att = count - 1

                idx_val_att = grp_indices[:n_val_att]
                idx_test_att = grp_indices[n_val_att:]

                if len(idx_val_att) > 0:
                    df_val_attacks_list.append(df.loc[idx_val_att])
                if len(idx_test_att) > 0:
                    df_test_attacks_list.append(df.loc[idx_test_att])

        df_val_att = pd.concat(df_val_attacks_list) if df_val_attacks_list else pd.DataFrame(columns=df.columns)
        df_test_att = pd.concat(df_test_attacks_list) if df_test_attacks_list else pd.DataFrame(columns=df.columns)

        # 4. Consolidar partições finais
        df_train = df_train_norm.copy()
        df_val = pd.concat([df_val_norm, df_val_att]).copy()
        df_test = pd.concat([df_test_norm, df_test_att]).copy()

        return {
            "train": df_train,
            "val": df_val,
            "test": df_test,
            "rare_classes": rare_classes,
        }
