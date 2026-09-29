import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from models.autoencoder import FCDAE
from data.dataset import TrafficParquetDataset


def get_best_device(requested_device: str = "auto") -> torch.device:
    if requested_device != "auto":
        if requested_device == "dml":
            try:
                import torch_directml
                print("Dispositivo: Utilizando aceleração por GPU via DirectML (AMD/Intel/NVIDIA).")
                return torch_directml.device()
            except ImportError:
                print("Aviso: torch-directml não encontrado. Utilizando CPU com suporte a multi-threading.")
                return torch.device("cpu")
        return torch.device(requested_device)

    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        print(f"Dispositivo: Utilizando aceleração GPU CUDA ({device_name}).")
        return torch.device("cuda")
    
    try:
        import torch_directml
        print("Dispositivo: Utilizando aceleração por GPU DirectML (AMD Radeon / Direct3D 12).")
        return torch_directml.device()
    except ImportError:
        pass

    num_threads = os.cpu_count() or 4
    torch.set_num_threads(num_threads)
    print(f"Dispositivo: GPU CUDA/DirectML não detectada no ambiente PyTorch atual.")
    print(f"Utilizando paralelismo de CPU ativado com {num_threads} threads de alto desempenho.")
    return torch.device("cpu")


def train_autoencoder(
    parquet_path: str,
    output_model_path: str = "models/fc_dae_best.pt",
    label_column: str = "Label",
    normal_value: str = "0",
    epochs: int = 100,
    batch_size: int = 2048,
    lr: float = 1e-3,
    patience: int = 10,
    bottleneck_dim: int = None,
    device_name: str = "auto",
    num_workers: int = 0,
    seed: int = 42
):
    print(f"Carregando tráfego normal de {parquet_path}...")
    torch.manual_seed(seed)
    if isinstance(normal_value, str) and normal_value.isdigit():
        normal_value = int(normal_value)

    full_dataset = TrafficParquetDataset(
        parquet_path=parquet_path,
        label_column=label_column,
        normal_label_value=normal_value,
        only_normal=True
    )

    n_samples = len(full_dataset)
    if n_samples < 2:
        raise ValueError(
            f"Amostras normais insuficientes para treinar o FC-DAE: {n_samples}. "
            f"Verifique 'label_column'/'normal_value' e o conteúdo de '{parquet_path}'."
        )

    # Garante ao menos uma amostra em cada lado da divisão, evitando divisão por zero
    # no cálculo das perdas médias quando o dataset é pequeno.
    val_size = min(max(1, int(n_samples * 0.2)), n_samples - 1)
    train_size = n_samples - val_size

    # Divisão semeada: sem isso a partição de validação muda a cada execução e as
    # perdas reportadas deixam de ser reproduzíveis.
    split_generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size], generator=split_generator)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(torch.cuda.is_available())
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(torch.cuda.is_available())
    )

    num_features = full_dataset.num_features
    print(f"Dataset carregado: {n_samples} amostras normais ({train_size} treino, {val_size} validação). Features: {num_features}")

    device = get_best_device(device_name)
    model = FCDAE(input_dim=num_features, bottleneck_dim=bottleneck_dim).to(device)
    print(f"Modelo FC-DAE criado. Input: {model.input_dim}, Bottleneck: {model.bottleneck_dim}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    best_val_loss = float('inf')
    epochs_no_improve = 0

    os.makedirs(os.path.dirname(output_model_path), exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for x_batch, _ in train_loader:
            x_batch = x_batch.to(device)
            optimizer.zero_grad()
            x_hat, _ = model(x_batch)
            loss = criterion(x_hat, x_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x_batch.size(0)

        train_loss /= train_size

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_batch, _ in val_loader:
                x_batch = x_batch.to(device)
                x_hat, _ = model(x_batch)
                loss = criterion(x_hat, x_batch)
                val_loss += loss.item() * x_batch.size(0)

        val_loss /= val_size
        scheduler.step(val_loss)

        print(f"Época {epoch:03d}/{epochs:03d} | Perda Treino: {train_loss:.6f} | Perda Val: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'input_dim': num_features,
                'bottleneck_dim': model.bottleneck_dim,
                'best_val_loss': best_val_loss,
                'feature_columns': full_dataset.feature_columns
            }, output_model_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Parada antecipada atingida ({patience} épocas sem melhora). Treinamento finalizado.")
                break

    print(f"Modelo otimizado salvo em '{output_model_path}' com menor perda de validação: {best_val_loss:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Treino do FC-DAE Módulo 2")
    parser.add_argument("--parquet_path", type=str, required=True, help="Caminho do arquivo .parquet de tráfego")
    parser.add_argument("--output_model_path", type=str, default="models/fc_dae_best.pt", help="Caminho de saída para os pesos do modelo")
    parser.add_argument("--label_column", type=str, default="Label", help="Nome da coluna de rótulo")
    parser.add_argument("--normal_value", type=str, default="0", help="Valor indicativo de classe benigna/normal")
    parser.add_argument("--epochs", type=int, default=100, help="Número máximo de épocas")
    parser.add_argument("--batch_size", type=int, default=2048, help="Tamanho do mini-batch (padrão: 2048)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Taxa de aprendizado inicial")
    parser.add_argument("--patience", type=int, default=10, help="Paciência para Early Stopping")
    parser.add_argument("--bottleneck_dim", type=int, default=None, help="Dimensão manual do bottleneck (opcional)")
    parser.add_argument("--device", type=str, default="auto", help="Dispositivo: auto, cpu, cuda, dml")
    parser.add_argument("--num_workers", type=int, default=0, help="Número de sub-processos no DataLoader")
    parser.add_argument("--seed", type=int, default=42, help="Semente da divisão treino/validação e dos pesos")

    args = parser.parse_args()
    train_autoencoder(
        parquet_path=args.parquet_path,
        output_model_path=args.output_model_path,
        label_column=args.label_column,
        normal_value=args.normal_value,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        bottleneck_dim=args.bottleneck_dim,
        device_name=args.device,
        num_workers=args.num_workers,
        seed=args.seed
    )
