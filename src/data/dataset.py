import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import List, Optional, Tuple, Union


class TrafficParquetDataset(Dataset):
    """
    Dataset PyTorch para leitura de arquivos .parquet contendo tráfego de rede pré-processado.
    Permite filtragem exclusiva de tráfego normal para treinamento não-supervisionado.
    """

    def __init__(
        self,
        parquet_path: str,
        label_column: str = "Label",
        normal_label_value: Union[str, int] = 0,
        only_normal: bool = False,
        feature_columns: Optional[List[str]] = None
    ):
        """
        :param parquet_path: Caminho do arquivo .parquet.
        :param label_column: Nome da coluna contendo os rótulos de classe.
        :param normal_label_value: Valor identificador de tráfego normal (ex: 0 ou "BENIGN").
        :param only_normal: Se True, filtra o DataFrame mantendo apenas o tráfego normal.
        :param feature_columns: Lista de colunas numéricas de entrada. Se None, infere excluindo a label_column.
        """
        import os
        if not os.path.exists(parquet_path):
            raise FileNotFoundError(
                f"O arquivo Parquet especificado não foi encontrado: '{os.path.abspath(parquet_path)}'. "
                f"Verifique o caminho do arquivo e certifique-se de que ele está correto."
            )

        df = pd.read_parquet(parquet_path)

        if only_normal and label_column in df.columns:
            df = df[df[label_column] == normal_label_value].reset_index(drop=True)

        if label_column in df.columns:
            self.labels = df[label_column].values
            if feature_columns is None:
                self.feature_columns = [col for col in df.columns if col != label_column]
            else:
                self.feature_columns = feature_columns
        else:
            self.labels = None
            if feature_columns is None:
                self.feature_columns = list(df.columns)
            else:
                self.feature_columns = feature_columns

        features_data = df[self.feature_columns].astype("float32").values
        self.features = torch.tensor(features_data, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Optional[Union[str, int]]]:
        label = self.labels[idx] if self.labels is not None else -1
        return self.features[idx], label

    @property
    def num_features(self) -> int:
        return len(self.feature_columns)
