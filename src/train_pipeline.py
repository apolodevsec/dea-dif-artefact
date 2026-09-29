"""Rotina completa de treino e validação do framework híbrido FC-DAE + Deep Isolation Forest.

Executa, para cada dataset processado (NSL-KDD, UNSW-NB15, CICIDS2017), a cadeia
integral dos Módulos 2 a 5:

1. **Particionamento (leakage-free)** — o teste fornecido no artefato permanece cego;
   toda calibração deriva de `X_train_baseline`.
2. **Módulo 2 — FC-DAE** — treino não-supervisionado no tráfego benigno, validado em
   partição benigna disjunta (early stopping + `ReduceLROnPlateau`).
3. **Módulo 3 — Pre-IF** — extração do espaço latente `z` e do MSE por fluxo;
   calibração de `tau_AE` no benigno de validação.
4. **Módulo 4 — DIF** — indução do ensemble estritamente sobre latentes benignos e
   calibração de `tau_DIF_ref` por percentil.
5. **Módulo 5 — Avaliação** — baselines B1..B6b no teste cego, recall estratificado
   por família de ataque, ablação do DEAS e sensibilidade OFAT opcional.

Uso típico:

```powershell
python -m src.train_pipeline --data_root "C:/caminho/para/os/datasets" --epochs 50
python -m src.train_pipeline --dataset_dir "D:/dados/NSL-KDD-processados" --run_ofat
```
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

if __package__ in (None, ""):  # permite execução direta via `python src/train_pipeline.py`
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.nids_datasets import (  # noqa: E402
    NidsPartitions,
    dataset_display_name,
    load_partitions,
    resolve_dataset_dirs,
)
from src.evaluation.baselines import BaselineRunner  # noqa: E402
from src.evaluation.reporting import build_consolidated_report, build_dataset_report  # noqa: E402
from src.evaluation.sensitivity import OFATSensitivityAnalyzer  # noqa: E402
from src.training.ae_trainer import (  # noqa: E402
    AETrainConfig,
    default_bottleneck_dim,
    save_training_history,
    train_fc_dae,
)
from src.training.calibration import (  # noqa: E402
    AE_THRESHOLD_RULES,
    calibrate_ae_thresholds,
    percentile_threshold,
)
from src.training.latent import extract_latent_and_mse, save_latent_parquet  # noqa: E402

# Fatores varridos pelo OFAT sobre o espaço latente já extraído. A dimensão m é
# omitida porque alterá-la exige retreinar o FC-DAE (ver `--ofat_m_values`).
DEFAULT_OFAT_SWEEPS: Dict[str, List[Any]] = {
    "L": [1, 2, 3, 4],
    "t": [50, 100, 200],
    "p": [90.0, 95.0, 97.0, 99.0],
    "lambda_deas": [0.0, 0.25, 0.5, 0.75, 1.0],
}


def _slugify(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name)


def _log_section(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def build_supervised_training_set(
    parts: NidsPartitions,
    max_rows: Optional[int],
    random_state: int,
) -> "tuple[np.ndarray, np.ndarray]":
    """Monta o conjunto rotulado do baseline supervisionado B4.

    Combina o benigno de treino com os ataques de treino reservados — nunca o teste
    cego. O teto `max_rows` é aplicado **antes** da concatenação, preservando a
    proporção entre as duas classes, para não materializar a pilha completa em
    datasets de milhões de fluxos.
    """
    n_normal = len(parts.X_ae_train)
    n_attack = len(parts.X_sup_attacks)
    total = n_normal + n_attack

    if max_rows is not None and 0 < max_rows < total:
        rng = np.random.RandomState(random_state)
        fraction = max_rows / total
        keep_normal = min(n_normal, max(1, int(round(n_normal * fraction))))
        keep_attack = min(n_attack, max(1 if n_attack else 0, int(round(n_attack * fraction))))
        X_normal = parts.X_ae_train[np.sort(rng.permutation(n_normal)[:keep_normal])]
        X_attack = (
            parts.X_sup_attacks[np.sort(rng.permutation(n_attack)[:keep_attack])]
            if keep_attack
            else parts.X_sup_attacks[:0]
        )
    else:
        X_normal, X_attack = parts.X_ae_train, parts.X_sup_attacks

    if len(X_attack) == 0:
        return X_normal, np.zeros(len(X_normal), dtype=int)

    X_sup = np.vstack([X_normal, X_attack])
    y_sup = np.concatenate([np.zeros(len(X_normal), dtype=int), np.ones(len(X_attack), dtype=int)])
    return X_sup, y_sup


def run_dataset(
    dataset_dir: str,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """Executa a cadeia completa de treino, validação e avaliação em um dataset."""
    name = dataset_display_name(dataset_dir)
    slug = _slugify(name)
    started = time.perf_counter()

    output_dir = os.path.join(args.output_dir, slug)
    models_dir = os.path.join(args.models_dir, slug)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Particionamento livre de vazamento
    # ------------------------------------------------------------------
    _log_section(f"[{name}] Etapa 1/5 — Particionamento leakage-free")
    parts: NidsPartitions = load_partitions(
        dataset_dir=dataset_dir,
        val_normal_ratio=args.val_normal_ratio,
        val_attack_ratio=args.val_attack_ratio,
        rare_threshold=args.rare_threshold,
        random_state=args.random_state,
        label_column=args.label_column,
        normal_label=args.normal_label,
        max_train_rows=args.max_train_rows,
        max_test_rows=args.max_test_rows,
        verbose=True,
    )

    # ------------------------------------------------------------------
    # 2. Módulo 2 — treino e validação do FC-DAE
    # ------------------------------------------------------------------
    _log_section(f"[{name}] Etapa 2/5 — Treino do FC-DAE (Módulo 2)")
    ae_config = AETrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        min_delta=args.min_delta,
        bottleneck_dim=args.bottleneck_dim,
        device=args.device,
        num_workers=args.num_workers,
        seed=args.random_state,
        verbose=True,
    )
    ae_result = train_fc_dae(
        X_train=parts.X_ae_train,
        X_val=parts.X_ae_val,
        config=ae_config,
        checkpoint_path=os.path.join(models_dir, "fc_dae_best.pt"),
        feature_columns=parts.feature_columns,
        extra_checkpoint_fields={
            "dataset": name,
            "label_column": parts.label_column,
            "normal_label": parts.normal_label,
        },
    )
    save_training_history(ae_result, os.path.join(output_dir, "fc_dae_training_history.json"))

    # ------------------------------------------------------------------
    # 3. Módulo 3 — espaço latente, MSE e calibração de tau_AE
    # ------------------------------------------------------------------
    _log_section(f"[{name}] Etapa 3/5 — Extração do espaço latente (Módulo 3)")
    Z_train, mse_train = extract_latent_and_mse(
        ae_result.model, parts.X_ae_train, ae_result.device, batch_size=args.inference_batch_size
    )
    Z_val, mse_val = extract_latent_and_mse(
        ae_result.model, parts.X_val, ae_result.device, batch_size=args.inference_batch_size
    )
    Z_test, mse_test = extract_latent_and_mse(
        ae_result.model, parts.X_test, ae_result.device, batch_size=args.inference_batch_size
    )
    print(
        f"[{name}] Latentes extraídos: treino {Z_train.shape}, validação {Z_val.shape}, teste {Z_test.shape}"
    )

    mse_val_normal = mse_val[parts.val_normal_mask]
    calibration: Dict[str, Any] = calibrate_ae_thresholds(
        mse_val_normal,
        k_sigma=args.k_sigma,
        percentile=args.ae_percentile,
        rule=args.ae_threshold_rule,
    )
    print(
        f"[{name}] tau_AE adotado (regra '{calibration['tau_ae_rule']}') = "
        f"{calibration['tau_ae_ref']:.6f}"
    )
    print(
        f"[{name}]   percentil p{args.ae_percentile:g} = {calibration['tau_ae_percentile']:.6f} "
        f"(FPR esperado {calibration['expected_fpr_percentile']:.4%}) | "
        f"mu+{args.k_sigma:g}sigma = {calibration['tau_ae_mu_sigma']:.6f} "
        f"(FPR esperado {calibration['expected_fpr_mu_sigma']:.4%}, "
        f"equivale ao percentil empírico {calibration['mu_sigma_empirical_percentile']:.2f})"
    )

    mse_attacks = mse_val[~parts.val_normal_mask]
    calibration["mse_val_attack_mean"] = float(mse_attacks.mean()) if mse_attacks.size else float("nan")
    calibration["mse_separation_ratio"] = (
        float(mse_attacks.mean() / max(calibration["mse_val_normal_mean"], 1e-12))
        if mse_attacks.size
        else float("nan")
    )

    if args.save_latent:
        save_latent_parquet(
            os.path.join(output_dir, "latent_space_test.parquet"),
            Z_test,
            mse_test,
            classes=parts.classes_test,
            label_column=parts.label_column,
        )
        save_latent_parquet(
            os.path.join(output_dir, "latent_space_val.parquet"),
            Z_val,
            mse_val,
            classes=parts.classes_val,
            label_column=parts.label_column,
        )
        print(f"[{name}] Espaços latentes persistidos em '{output_dir}'.")

    # ------------------------------------------------------------------
    # 4/5. Módulos 4 e 5 — DIF, baselines e avaliação no teste cego
    # ------------------------------------------------------------------
    _log_section(f"[{name}] Etapa 4/5 — DIF e baselines (Módulos 4 e 5)")

    projection_dim = args.projection_dim or ae_result.bottleneck_dim
    runner = BaselineRunner(
        n_estimators=args.trees,
        max_samples=args.subsample,
        n_representations=args.representations,
        n_layers=args.n_layers,
        projection_dim=projection_dim,
        lambda_deas=args.lambda_deas,
        p_low=args.p_low,
        p_high=args.p_high,
        n_grid=args.alpha_grid,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
        score_batch_size=args.score_batch_size,
        max_supervised_rows=args.max_supervised_rows,
    )

    X_sup_train, y_sup_train = build_supervised_training_set(parts, args.max_supervised_rows, args.random_state)

    base_results = runner.run_all(
        X_train_raw=parts.X_ae_train,
        Z_train=Z_train,
        mse_train=mse_train,
        X_val_raw=parts.X_val,
        Z_val=Z_val,
        mse_val=mse_val,
        y_val=parts.y_val,
        X_test_raw=parts.X_test,
        Z_test=Z_test,
        mse_test=mse_test,
        y_test=parts.y_test,
        test_classes=parts.classes_test,
        val_normal_mask=parts.val_normal_mask,
        X_sup_train=X_sup_train,
        y_sup_train=y_sup_train,
    )
    del X_sup_train, y_sup_train

    summary_df: pd.DataFrame = base_results["summary_df"]
    dif_model = base_results["models"]["b6_dif"]

    # tau_DIF_ref: limiar não-supervisionado por percentil sobre o benigno de validação.
    scores = base_results.get("scores")
    if scores is not None:
        dif_val_deas = np.asarray(scores["dif_val_deas"])[parts.val_normal_mask]
        dif_val_standard = np.asarray(scores["dif_val_standard"])[parts.val_normal_mask]
    else:  # compatibilidade: recalcula caso o runner não exponha os escores
        rescored = dif_model.score_samples(Z_val[parts.val_normal_mask], batch_size=args.score_batch_size)
        dif_val_deas = rescored["score_dif_deas"]
        dif_val_standard = rescored["score_dif_standard"]

    dif_report: Dict[str, Any] = {
        "n_estimators": args.trees,
        "max_samples": args.subsample,
        "n_representations": min(args.representations, args.trees),
        "n_layers": args.n_layers,
        "projection_dim": projection_dim,
        "lambda_deas": args.lambda_deas,
        "val_percentile": args.val_percentile,
        "tau_dif_ref_deas": percentile_threshold(dif_val_deas, args.val_percentile),
        "tau_dif_ref_standard": percentile_threshold(dif_val_standard, args.val_percentile),
        "fit_seconds": float(base_results.get("dif_fit_seconds", float("nan"))),
        "shared_forest_for_ablation": bool(base_results.get("shared_dif_forest", True)),
        "b4_supervised_rows": int(base_results.get("b4_supervised_rows", 0)),
    }
    print(
        f"[{name}] tau_DIF_ref (DEAS, p{args.val_percentile:g}) = {dif_report['tau_dif_ref_deas']:.6f}"
    )

    calibration["supervised"] = {k: float(v) for k, v in base_results["thresholds"].items()}

    # Escores por baseline no teste cego: insumo das curvas ROC/PR do Capítulo 5.
    if args.save_scores:
        per_model = base_results.get("test_scores_by_model", {})
        scores_frame = pd.DataFrame({"y_true": parts.y_test, parts.label_column: parts.classes_test})
        for model_name, values in per_model.items():
            scores_frame[model_name] = values
        scores_path = os.path.join(output_dir, "test_scores.parquet")
        scores_frame.to_parquet(scores_path, index=False)
        print(f"[{name}] Escores do teste por baseline gravados em '{scores_path}'.")

    print()
    print(f"[{name}] Tabela 4 — métricas no teste cego:")
    display_cols = [c for c in ["accuracy", "precision", "recall", "f1_score", "auc_roc", "fpr"] if c in summary_df]
    print(summary_df[display_cols].to_string(float_format=lambda v: f"{v:.4f}"))

    # ------------------------------------------------------------------
    # OFAT opcional
    # ------------------------------------------------------------------
    ofat_results: Dict[str, Any] = {}
    if args.run_ofat:
        _log_section(f"[{name}] Etapa 5/5 — Sensibilidade OFAT")
        analyzer = OFATSensitivityAnalyzer(
            base_params={
                "m": ae_result.bottleneck_dim,
                "L": args.n_layers,
                "t": args.trees,
                "p": args.val_percentile,
                "lambda_deas": args.lambda_deas,
                "n_representations": args.representations,
            },
            random_state=args.random_state,
            n_jobs=args.n_jobs,
        )
        sweeps = analyzer.run_sweeps(
            X_train=Z_train,
            X_val=Z_val,
            y_val=parts.y_val,
            X_test=Z_test,
            y_test=parts.y_test,
            sweeps=DEFAULT_OFAT_SWEEPS,
            output_figure_dir=os.path.join(output_dir, "figures", "sensitivity"),
        )
        ofat_results = {
            "sweeps": {k: v.to_dict(orient="records") for k, v in sweeps.items()},
            "metadata": analyzer.get_metadata(),
        }

    if args.ofat_m_values:
        _log_section(f"[{name}] Sensibilidade à dimensão latente m (com retreino do FC-DAE)")
        ofat_results["m_retrained"] = run_latent_dim_sweep(parts, args, output_dir)

    # ------------------------------------------------------------------
    # Relatórios
    # ------------------------------------------------------------------
    rule = calibration["tau_ae_rule"]
    notes = [
        "O conjunto de teste fornecido no artefato foi usado uma única vez, na avaliação "
        "final; nenhum limiar, alfa ou hiperparâmetro foi ajustado sobre ele.",
        "Os ataques da validação provêm de `X_train_baseline`, mantendo o teste cego, "
        "conforme o particionamento estratificado com ataques na validação.",
        f"`tau_AE` adotado pela regra `{rule}` "
        + (
            f"(percentil p={args.ae_percentile:g} do MSE benigno de validação, Eq. 4 / Seção 4.7.1)."
            if rule == "percentile"
            else f"(mu + {args.k_sigma:g}*sigma do MSE benigno de validação, formulação de `ae.md`)."
        )
        + " As duas formulações são reportadas lado a lado porque não são equivalentes: "
        f"neste dataset `mu + {args.k_sigma:g}*sigma` cai no percentil empírico "
        f"{calibration['mu_sigma_empirical_percentile']:.2f}, não em 99.",
        "`tau_AE` e `tau_DIF_ref` são limiares não-supervisionados (só tráfego benigno). "
        "Os limiares de Youden e o alfa da fusão são supervisionados e constituem o teto "
        "de operação calibrado na validação rotulada.",
    ]

    temporal = parts.meta.get("temporal_support", {})
    if not temporal.get("b5_lstm_executable", False):
        notes.append(
            "O baseline B5 (LSTM-AE) **não** é executado. O janelamento cronológico "
            "(W=10, passo S=1, rótulo em t_W) exige a coluna `Timestamp`, e o artefato "
            "processado não a contém: nenhuma das features é temporal e não há arquivo "
            "paralelo de metadados/timestamps no diretório do dataset. Para destravar B5 "
            "é preciso reexecutar o Módulo 1 preservando a coluna temporal original."
        )

    provenance = parts.meta.get("split_provenance", {})
    if provenance:
        notes.append(
            f"Procedência do split treino/teste do artefato: {provenance['verdict']} "
            f"(teste = {provenance['test_fraction']:.2%} do total; maior divergência de "
            f"proporção de classe entre treino e teste = "
            f"{provenance['max_class_proportion_delta_pp']:.4f} p.p.; famílias exclusivas "
            f"do teste = {provenance['families_only_in_test'] or 'nenhuma'})."
        )

    share = parts.meta.get("global_partition_share", {})
    if share:
        notes.append(
            "Particionamento efetivo sobre o total de cada classe — benigno "
            f"{share['normal']['train_unsupervised']:.1%} treino / "
            f"{share['normal']['validation']:.1%} validação / "
            f"{share['normal']['test']:.1%} teste; ataques "
            f"{share['attack']['validation']:.1%} validação / "
            f"{share['attack']['reserved_supervised']:.1%} reservados ao B4 / "
            f"{share['attack']['test']:.1%} teste. Difere do esquema nominal 70/15/15 "
            "porque a fronteira treino/teste é fixada pelo artefato e o teste permanece "
            "intocado; reparticionar para 70/15/15 exigiria reslicing do teste."
        )

    if parts.meta.get("autoencoder_file_check", {}).get("row_count_matches"):
        notes.append(
            "Consistência verificada: `X_train_autoencoder.parquet` tem exatamente o mesmo "
            "número de linhas do subconjunto benigno de `X_train_baseline.parquet`."
        )

    elapsed = time.perf_counter() - started
    result: Dict[str, Any] = {
        "dataset": name,
        "dataset_dir": os.path.abspath(dataset_dir),
        "partitions": parts.meta,
        "autoencoder": ae_result.to_report(),
        "dif": dif_report,
        "calibration": calibration,
        "baselines": summary_df.to_dict(orient="index"),
        "stratified_recall": base_results["stratified_recall"],
        "delta_deas_f1": base_results["delta_deas_f1"],
        "delta_deas_auc": base_results["delta_deas_auc"],
        "rare_classes": parts.rare_classes,
        "ofat": ofat_results,
        "wall_seconds": elapsed,
    }

    markdown = build_dataset_report(
        dataset_name=name,
        partition_meta=parts.meta,
        ae_report=ae_result.to_report(),
        dif_report=dif_report,
        calibration=calibration,
        summary_df=summary_df,
        stratified_recall=base_results["stratified_recall"],
        delta_deas_f1=base_results["delta_deas_f1"],
        delta_deas_auc=base_results["delta_deas_auc"],
        rare_classes=parts.rare_classes,
        ofat=ofat_results,
        notes=notes,
    )

    md_path = os.path.join(output_dir, "experimental_results.md")
    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write(markdown)

    json_path = os.path.join(output_dir, "experimental_results.json")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False, default=str)

    print()
    print(f"[{name}] Concluído em {elapsed:.1f}s. Relatórios em '{output_dir}'.")
    return result


def run_latent_dim_sweep(
    parts: NidsPartitions,
    args: argparse.Namespace,
    output_dir: str,
) -> List[Dict[str, Any]]:
    """Varredura da dimensão latente `m` com retreino completo do FC-DAE por valor.

    Diferente da varredura OFAT sobre o latente já extraído, alterar `m` muda a
    própria representação: cada valor exige um novo FC-DAE, nova extração latente e
    nova indução do DIF. Por isso é opt-in via `--ofat_m_values`.
    """
    from src.evaluation.evaluator import ExperimentalEvaluator
    from src.evaluation.hybrid import _find_best_youden_threshold
    from src.models.dif import DeepIsolationForest

    evaluator = ExperimentalEvaluator()
    records: List[Dict[str, Any]] = []

    for m_value in args.ofat_m_values:
        m_value = int(m_value)
        if not 1 <= m_value <= parts.n_features:
            print(f"  m={m_value} ignorado (fora de [1, {parts.n_features}]).")
            continue

        print(f"  Retreinando FC-DAE com m={m_value}...")
        config = AETrainConfig(
            epochs=args.ofat_m_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            patience=max(3, args.patience // 2),
            bottleneck_dim=m_value,
            device=args.device,
            num_workers=args.num_workers,
            seed=args.random_state,
            verbose=False,
        )
        ae = train_fc_dae(parts.X_ae_train, parts.X_ae_val, config=config)

        Z_tr, _ = extract_latent_and_mse(ae.model, parts.X_ae_train, ae.device, args.inference_batch_size)
        Z_va, _ = extract_latent_and_mse(ae.model, parts.X_val, ae.device, args.inference_batch_size)
        Z_te, _ = extract_latent_and_mse(ae.model, parts.X_test, ae.device, args.inference_batch_size)

        dif = DeepIsolationForest(
            n_estimators=args.trees,
            max_samples=args.subsample,
            n_representations=args.representations,
            n_layers=args.n_layers,
            lambda_deas=args.lambda_deas,
            random_state=args.random_state,
            n_jobs=args.n_jobs,
        )
        dif.fit(Z_tr)

        scores_val = dif.score_samples(Z_va, batch_size=args.score_batch_size)["score_dif_deas"]
        tau, _ = _find_best_youden_threshold(parts.y_val, scores_val)

        t0 = time.perf_counter()
        scores_test = dif.score_samples(Z_te, batch_size=args.score_batch_size)["score_dif_deas"]
        latency = ((time.perf_counter() - t0) * 1000.0) / max(1, len(parts.y_test))

        metrics = evaluator.evaluate(
            parts.y_test,
            (scores_test > tau).astype(int),
            scores=scores_test,
            latency_ms_per_sample=latency,
        )
        record = {
            "param_name": "m",
            "param_value": m_value,
            "tau": float(tau),
            "ae_best_val_loss": ae.best_val_loss,
            **metrics,
        }
        records.append(record)
        print(
            f"  m={m_value}: MSE val={ae.best_val_loss:.6f} | F1={metrics['f1_score']:.4f} "
            f"| AUC={metrics['auc_roc']:.4f}"
        )

    if records:
        path = os.path.join(output_dir, "ofat_latent_dim_sweep.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(records, handle, indent=2, ensure_ascii=False)

    return records


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Treino e validação do framework híbrido FC-DAE + Deep Isolation Forest",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    source = parser.add_argument_group("Fonte dos dados")
    source.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Diretório que contém os subdiretórios de dataset processados",
    )
    source.add_argument(
        "--dataset_dir",
        type=str,
        action="append",
        default=None,
        help="Diretório de um dataset específico (pode ser repetido)",
    )
    source.add_argument(
        "--datasets",
        type=str,
        default=None,
        help="Lista separada por vírgula para filtrar os datasets em --data_root (ex.: NSL-KDD,CICIDS2017)",
    )
    source.add_argument("--label_column", type=str, default=None, help="Força o nome da coluna de rótulo")
    source.add_argument("--normal_label", type=str, default=None, help="Força o valor do rótulo benigno")

    outputs = parser.add_argument_group("Saídas")
    outputs.add_argument("--output_dir", type=str, default="reports", help="Diretório de relatórios")
    outputs.add_argument("--models_dir", type=str, default="models", help="Diretório dos checkpoints")
    outputs.add_argument(
        "--save_latent", action="store_true", help="Persiste os espaços latentes de validação e teste em parquet"
    )
    outputs.add_argument(
        "--save_scores",
        action="store_true",
        help="Persiste os escores de cada baseline no teste cego (insumo das curvas ROC/PR)",
    )

    protocol = parser.add_argument_group("Protocolo de particionamento")
    protocol.add_argument("--val_normal_ratio", type=float, default=0.15, help="Fração benigna de validação")
    protocol.add_argument("--val_attack_ratio", type=float, default=0.20, help="Fração de ataques na validação")
    protocol.add_argument("--rare_threshold", type=int, default=15, help="Limite de instâncias para classe rara")
    protocol.add_argument("--random_state", type=int, default=42, help="Semente determinística global")
    protocol.add_argument("--max_train_rows", type=int, default=None, help="Subamostragem estratificada do treino")
    protocol.add_argument("--max_test_rows", type=int, default=None, help="Subamostragem estratificada do teste")

    autoencoder = parser.add_argument_group("Módulo 2 — FC-DAE")
    autoencoder.add_argument("--epochs", type=int, default=50, help="Número máximo de épocas")
    autoencoder.add_argument("--batch_size", type=int, default=2048, help="Tamanho do mini-batch")
    autoencoder.add_argument("--lr", type=float, default=1e-3, help="Taxa de aprendizado inicial")
    autoencoder.add_argument("--weight_decay", type=float, default=0.0, help="Regularização L2 do Adam")
    autoencoder.add_argument("--patience", type=int, default=10, help="Paciência do early stopping")
    autoencoder.add_argument("--min_delta", type=float, default=0.0, help="Melhora mínima para reiniciar a paciência")
    autoencoder.add_argument("--bottleneck_dim", type=int, default=None, help="Dimensão m (padrão: floor(n/8))")
    autoencoder.add_argument("--device", type=str, default="auto", help="auto, cpu, cuda, cuda:0 ou dml")
    autoencoder.add_argument("--num_workers", type=int, default=0, help="Workers do DataLoader")
    autoencoder.add_argument("--inference_batch_size", type=int, default=16384, help="Batch da extração latente")
    autoencoder.add_argument(
        "--ae_threshold_rule",
        type=str,
        default="percentile",
        choices=list(AE_THRESHOLD_RULES),
        help="Formulação de tau_AE: percentil p (Eq. 4 do TCC) ou mu + k*sigma (ae.md)",
    )
    autoencoder.add_argument("--ae_percentile", type=float, default=95.0, help="Percentil p de tau_AE")
    autoencoder.add_argument("--k_sigma", type=float, default=3.0, help="k de tau_AE = mu + k*sigma")

    dif = parser.add_argument_group("Módulo 4 — Deep Isolation Forest")
    dif.add_argument("--trees", type=int, default=100, help="Árvores de isolamento (t)")
    dif.add_argument("--subsample", type=int, default=256, help="Subamostra por árvore (psi)")
    dif.add_argument("--representations", type=int, default=10, help="Redes de projeção aleatória (r)")
    dif.add_argument("--n_layers", type=int, default=3, help="Camadas por rede Phi_i (L)")
    dif.add_argument("--projection_dim", type=int, default=None, help="Dimensão de projeção d (padrão: m)")
    dif.add_argument("--lambda_deas", type=float, default=0.5, help="Fator de ponderação DEAS")
    dif.add_argument("--val_percentile", type=float, default=95.0, help="Percentil de tau_DIF_ref")

    evaluation = parser.add_argument_group("Módulo 5 — Avaliação")
    evaluation.add_argument("--p_low", type=float, default=1.0, help="Percentil inferior da normalização do MSE")
    evaluation.add_argument("--p_high", type=float, default=99.0, help="Percentil superior da normalização do MSE")
    evaluation.add_argument("--alpha_grid", type=int, default=101, help="Pontos da grade de alfa da fusão linear")
    evaluation.add_argument(
        "--max_supervised_rows",
        type=int,
        default=300_000,
        help="Teto de linhas no treino do baseline supervisionado B4 (0 desativa)",
    )
    evaluation.add_argument("--score_batch_size", type=int, default=32768, help="Bloco de inferência do DIF")
    evaluation.add_argument("--n_jobs", type=int, default=-1, help="Paralelismo de scikit-learn e joblib")
    evaluation.add_argument("--run_ofat", action="store_true", help="Executa a sensibilidade OFAT (L, t, p, lambda)")
    evaluation.add_argument(
        "--ofat_m_values",
        type=int,
        nargs="*",
        default=None,
        help="Valores de m para varredura com retreino do FC-DAE (custoso)",
    )
    evaluation.add_argument("--ofat_m_epochs", type=int, default=15, help="Épocas por retreino na varredura de m")

    return parser


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.max_supervised_rows is not None and args.max_supervised_rows <= 0:
        args.max_supervised_rows = None

    if args.dataset_dir:
        dataset_dirs = [os.path.abspath(d) for d in args.dataset_dir]
        for directory in dataset_dirs:
            if not os.path.isdir(directory):
                parser.error(f"Diretório de dataset inexistente: '{directory}'")
    elif args.data_root:
        names = [n.strip() for n in args.datasets.split(",")] if args.datasets else None
        dataset_dirs = resolve_dataset_dirs(args.data_root, names)
    else:
        parser.error("Informe --data_root ou --dataset_dir.")

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Datasets selecionados ({len(dataset_dirs)}):")
    for directory in dataset_dirs:
        print(f"  - {directory}")

    results: List[Dict[str, Any]] = []
    failures: List[Dict[str, str]] = []

    for directory in dataset_dirs:
        try:
            results.append(run_dataset(directory, args))
        except Exception as error:  # um dataset com problema não aborta os demais
            print(f"\n[ERRO] Falha ao processar '{directory}': {type(error).__name__}: {error}")
            failures.append({"dataset_dir": directory, "error": f"{type(error).__name__}: {error}"})

    payload: Dict[str, Any] = {
        "datasets": results,
        "failures": failures,
        "config": vars(args),
    }

    if results:
        consolidated_md = os.path.join(args.output_dir, "consolidated_results.md")
        with open(consolidated_md, "w", encoding="utf-8") as handle:
            handle.write(build_consolidated_report(results))

        consolidated_json = os.path.join(args.output_dir, "consolidated_results.json")
        with open(consolidated_json, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False, default=str)

        _log_section("Consolidação")
        print(f"Datasets concluídos: {len(results)} | falhas: {len(failures)}")
        print(f"Relatório consolidado: '{consolidated_md}'")

    if failures and not results:
        raise SystemExit(1)

    return payload


if __name__ == "__main__":
    main()
