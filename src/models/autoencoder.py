import torch
import torch.nn as nn
from typing import Tuple, Optional


class FCDAE(nn.Module):
    """
    Fully Connected Deep Autoencoder (FC-DAE) para o Módulo 2 do framework de NIDS.
    
    Arquitetura simétrica:
    - Encoder:  n -> n/2 -> n/4 -> m (Ativação linear no bottleneck)
    - Decoder:  m -> n/4 -> n/2 -> n (Ativação Sigmoide na saída)
    """

    def __init__(self, input_dim: int, bottleneck_dim: Optional[int] = None):
        super(FCDAE, self).__init__()
        
        self.input_dim = input_dim
        
        # Define dimensão do bottleneck (m) caso não seja especificada expressamente
        if bottleneck_dim is None:
            if input_dim < 50:
                self.bottleneck_dim = max(1, input_dim // 4)
            else:
                self.bottleneck_dim = max(1, input_dim // 8)
        else:
            self.bottleneck_dim = bottleneck_dim

        hidden1 = max(1, input_dim // 2)
        hidden2 = max(1, input_dim // 4)

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, hidden1),
            nn.ReLU(inplace=True),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden2, self.bottleneck_dim)  # Ativação Linear preserva continuidade
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.bottleneck_dim, hidden2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden2, hidden1),
            nn.ReLU(inplace=True),
            nn.Linear(hidden1, self.input_dim),
            nn.Sigmoid()  # Saída rescada em [0, 1] compatível com Min-Max
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Extrai o vetor de representação latente comprimido (z)."""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstrói o vetor de entrada original a partir de z."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Passagem direta. Retorna (x_reconstruido, z_latente).
        """
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z
