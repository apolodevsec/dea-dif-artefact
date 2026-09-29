import json
import re

import numpy as np
import pandas as pd
import pytest

from src.evaluation import figures, tables
from src.make_tcc_assets import discover_results, main


# --------------------------------------------------------------------------
# Fixture: um reports/ sintético no formato que a rotina de treino produz
# --------------------------------------------------------------------------
def _baseline_metrics(seed: int) -> dict:
    rng = np.random.RandomState(seed)
    return {
        "accuracy": float(rng.uniform(0.7, 0.99)),
        "precision": float(rng.uniform(0.6, 0.99)),
        "recall": float(rng.uniform(0.6, 0.99)),
        "f1_score": float(rng.uniform(0.6, 0.99)),
        "fpr": float(rng.uniform(0.001, 0.2)),
        "auc_roc": float(rng.uniform(0.7, 0.999)),
        "latency_ms_per_sample": float(rng.uniform(0.001, 0.1)),
        "tp": 100, "tn": 100, "fp": 5, "fn": 5,
    }


def write_report(
    reports_dir,
    dataset: str,
    label_column: str,
    normal_label: str,
    families: dict,
    with_scores: bool = True,
    with_ofat: bool = True,
    stale: bool = False,
    subsampled: bool = False,
):
    """Materializa um `experimental_results.json` (e artefatos opcionais) de um dataset."""
    directory = reports_dir / dataset
    directory.mkdir(parents=True, exist_ok=True)

    models = list(tables.MODEL_ORDER)
    baselines = {m: _baseline_metrics(i) for i, m in enumerate(models)}
    stratified = {
        m: {
            family: {
                "count": count,
                "detected": int(count * 0.8),
                "recall": 0.8,
            }
            for family, count in families.items()
        }
        for m in models
    }

    payload = {
        "dataset": dataset,
        "partitions": {
            "label_column": label_column,
            "normal_label": normal_label,
            "n_features": 77,
            "protocol": {
                "val_normal_ratio": 0.15,
                "val_attack_ratio": 0.20,
                "rare_threshold": 15,
                "random_state": 42,
                "max_train_rows": 50000 if subsampled else None,
                "max_test_rows": None,
            },
            "sizes": {
                "ae_train_normal": 1000, "ae_val_normal": 200, "val_total": 300,
                "val_attacks": 100, "sup_train_attacks": 400, "test_total": 500,
                "test_normal": 300, "test_attacks": 200,
            },
        },
        "autoencoder": {
            "input_dim": 77,
            "bottleneck_dim": 9,
            "best_val_loss": 0.0031,
            "best_epoch": 4,
            "epochs_run": 5,
            "early_stopped": False,
            "train_seconds": 12.3,
            "device": "cpu",
            "history": [
                {"epoch": e, "train_loss": 0.1 / e, "val_loss": 0.11 / e, "lr": 1e-3}
                for e in range(1, 6)
            ],
        },
        "dif": {
            "n_estimators": 100, "max_samples": 256, "n_representations": 10,
            "n_layers": 3, "projection_dim": 9, "lambda_deas": 0.5,
            "val_percentile": 95.0, "tau_dif_ref_deas": 0.65,
            "tau_dif_ref_standard": 0.59, "fit_seconds": 0.8,
        },
        "calibration": {
            "percentile": 95.0, "k_sigma": 3.0,
            "tau_ae_percentile": 0.0239, "tau_ae_mu_sigma": 0.0310,
            "tau_ae_ref": 0.0239,
            "supervised": {"B6a_Hybrid_Linear_alpha": 0.72, "B6a_Hybrid_Linear_tau": 0.51},
        },
        "baselines": baselines,
        "stratified_recall": stratified,
        "delta_deas_f1": 0.0621,
        "delta_deas_auc": 0.0682,
        "rare_classes": {f: c for f, c in families.items() if c < 15},
        "ofat": {},
        "wall_seconds": 140.0,
    }

    if not stale:
        payload["calibration"]["tau_ae_rule"] = "percentile"
        payload["partitions"]["split_provenance"] = {
            "test_fraction": 0.30,
            "max_class_proportion_delta_pp": 0.0009,
            "families_only_in_test": [],
            "preserves_novel_attack_scenario": False,
            "looks_like_stratified_resplit": True,
            "verdict": "Refatiamento aleatório estratificado.",
        }
        payload["partitions"]["global_partition_share"] = {
            "normal": {"train_unsupervised": 0.595, "validation": 0.105, "test": 0.30, "total": 1500},
            "attack": {"validation": 0.14, "reserved_supervised": 0.56, "test": 0.30, "total": 700},
        }

    if with_ofat:
        payload["ofat"] = {
            "sweeps": {
                factor: [
                    {
                        "param_name": factor, "param_value": value, "tau": 0.5,
                        **_baseline_metrics(i),
                    }
                    for i, value in enumerate(values)
                ]
                for factor, values in (("L", [1, 2, 3]), ("t", [50, 100]), ("p", [90.0, 95.0]),
                                       ("lambda_deas", [0.0, 0.5, 1.0]))
            },
            "metadata": {"methodology_notes": "second_order_interactions não mapeadas."},
        }

    with open(directory / "experimental_results.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle)

    if with_scores:
        rng = np.random.RandomState(7)
        n = 500
        y_true = np.array([0] * 300 + [1] * 200)
        frame = pd.DataFrame({"y_true": y_true})
        classes = [normal_label] * 300
        for family, count in families.items():
            classes += [family] * count
        classes = (classes + [list(families)[0]] * n)[:n]
        frame[label_column] = classes
        for m in models:
            base = rng.uniform(0.0, 0.4, size=n)
            frame[m] = base + y_true * rng.uniform(0.2, 0.5, size=n)
        frame.to_parquet(directory / "test_scores.parquet", index=False)

    return directory


@pytest.fixture()
def reports(tmp_path):
    root = tmp_path / "reports"
    write_report(root, "NSL-KDD", "xAttack", "Normal", {"DoS": 200, "Probe": 60, "U2R": 4})
    write_report(root, "UNSW-NB15", "attack_cat", "Normal", {"Exploits": 150, "Worms": 5})
    return root


# --------------------------------------------------------------------------
# Paleta e conformidade com o método de dataviz
# --------------------------------------------------------------------------
def test_palette_slots_are_the_validated_reference_order():
    """A ordem categórica é o mecanismo de segurança CVD: não reordenar nem ciclar."""
    assert figures.SERIES[:4] == ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
    assert len(figures.SERIES) == 8
    # Curvas ROC sobrepostas ficam no limite de 3 slots validado em --pairs all.
    assert len(figures.ROC_ABLATION_CHAIN) == 3
    # O teto supervisionado é referência, não identidade: não consome slot categórico.
    assert figures.ROC_REFERENCE not in figures.ROC_ABLATION_CHAIN


def test_sequential_ramp_is_single_hue_and_monotonic():
    """Magnitude usa uma única matiz, clara -> escura. Nunca arco-íris."""
    from matplotlib.colors import to_rgb

    luminance = [sum(to_rgb(c)) / 3 for c in figures.SEQUENTIAL_BLUE]
    assert luminance == sorted(luminance, reverse=True)
    assert len(figures.SEQUENTIAL_BLUE) >= 8


def test_series_identity_is_never_colour_alone():
    """Cada slot tem marcador e estilo de linha: sobrevive a impressão em cinza."""
    assert len(figures.MARKERS) == len(figures.SERIES)
    assert len(figures.LINESTYLES) >= 4
    assert len(set(figures.MARKERS)) == len(figures.MARKERS)


def test_ofat_metrics_stay_within_four_series_on_one_axis():
    """Quatro métricas, todas em [0, 1], num único eixo — nunca eixo duplo."""
    assert len(figures.OFAT_METRICS) == 4


# --------------------------------------------------------------------------
# Geração de figuras
# --------------------------------------------------------------------------
def test_ae_convergence_figure_is_written(tmp_path):
    history = [{"epoch": e, "train_loss": 0.1 / e, "val_loss": 0.11 / e, "lr": 1e-3} for e in range(1, 8)]
    path = tmp_path / "conv.png"
    result = figures.plot_ae_convergence(history, "NSL-KDD", str(path))
    assert path.exists() and path.stat().st_size > 5000
    assert result.endswith("conv.png")


def test_roc_figure_handles_missing_models(tmp_path):
    rng = np.random.RandomState(0)
    y_true = np.array([0] * 80 + [1] * 40)
    scores = {"B1_iForest_Raw": rng.uniform(size=120) + y_true * 0.4}
    path = tmp_path / "roc.png"
    figures.plot_roc_curves(y_true, scores, "NSL-KDD", str(path))
    assert path.exists()


def test_mse_distribution_handles_zeros_and_threshold(tmp_path):
    rng = np.random.RandomState(1)
    mse = np.concatenate([np.zeros(5), rng.lognormal(-6, 1, 400), rng.lognormal(-3, 1, 100)])
    y_true = np.array([0] * 405 + [1] * 100)
    path = tmp_path / "mse.png"
    figures.plot_mse_distribution(mse, y_true, "NSL-KDD", str(path), tau_ae=0.01)
    assert path.exists()


def test_heatmap_marks_rare_families_with_ascii(tmp_path):
    """O marcador é ASCII: U+26A0 não existe nas fontes usadas e sairia como caixa."""
    stratified = {
        "B6a_Hybrid_Linear": {
            "DoS": {"count": 200, "recall": 0.95},
            "U2R": {"count": 4, "recall": 0.5},
        }
    }
    path = tmp_path / "heat.png"
    figures.plot_stratified_recall_heatmap(
        stratified, "NSL-KDD", str(path), rare_classes={"U2R": 4}
    )
    assert path.exists()

    # O marcador desenhado precisa existir nas fontes usadas pelo matplotlib; o
    # U+26A0 dos relatórios Markdown sairia como retângulo vazio na figura.
    assert figures.RARE_MARKER.isascii()
    assert figures.RARE_MARKER.strip() == "*"


def test_cross_dataset_comparison_rejects_more_datasets_than_slots(tmp_path):
    fake = [
        {"dataset": f"D{i}", "baselines": {m: {"f1_score": 0.5} for m in tables.MODEL_ORDER}}
        for i in range(len(figures.SERIES) + 1)
    ]
    with pytest.raises(ValueError, match="slots categóricos"):
        figures.plot_cross_dataset_comparison(fake, str(tmp_path / "x.png"))


# --------------------------------------------------------------------------
# Tabelas
# --------------------------------------------------------------------------
def test_latex_and_markdown_tables_carry_the_same_numbers(reports, tmp_path):
    with open(reports / "NSL-KDD" / "experimental_results.json", encoding="utf-8") as handle:
        result = json.load(handle)

    paths = tables.emit_table4(result, str(tmp_path))
    tex = open(paths["tex"], encoding="utf-8").read()
    md = open(paths["md"], encoding="utf-8").read()

    assert r"\begin{table}" in tex and r"\label{tab:tabela4-nsl-kdd}" in tex
    assert "| Modelo |" in md

    f1 = result["baselines"]["B6a_Hybrid_Linear"]["f1_score"]
    expected = f"{f1 * 100:.2f}"
    assert expected in tex and expected in md


def test_latex_escaping_protects_special_characters(tmp_path):
    result = {
        "dataset": "X",
        "baselines": {"B1_iForest_Raw": _baseline_metrics(0)},
        "stratified_recall": {
            "B1_iForest_Raw": {"Web Attack - Brute_Force & co": {"count": 20, "recall": 0.5}}
        },
        "rare_classes": {},
    }
    paths = tables.emit_stratified_recall(result, str(tmp_path))
    tex = open(paths["tex"], encoding="utf-8").read()
    assert r"Brute\_Force" in tex
    assert r"\&" in tex
    # Nenhum '_' ou '&' cru fora de comando LaTeX nas linhas de dados.
    for line in tex.splitlines():
        if line.startswith("Web Attack"):
            assert not re.search(r"(?<!\\)_", line)


def test_calibration_table_reports_both_tau_formulations(reports, tmp_path):
    with open(reports / "NSL-KDD" / "experimental_results.json", encoding="utf-8") as handle:
        result = json.load(handle)
    paths = tables.emit_calibration(result, str(tmp_path))
    tex = open(paths["tex"], encoding="utf-8").read()
    assert "0.023900" in tex  # percentil p=95
    assert "0.031000" in tex  # mu + 3 sigma
    assert "percentile" in tex


# --------------------------------------------------------------------------
# CLI ponta a ponta
# --------------------------------------------------------------------------
def test_asset_bundle_is_generated_end_to_end(reports, tmp_path):
    out = tmp_path / "assets"
    payload = main(["--reports_dir", str(reports), "--out_dir", str(out)])

    assert payload["assets"]
    manifest = (out / "MANIFEST.md").read_text(encoding="utf-8")
    assert "Capítulo 4" in manifest and "Capítulo 5" in manifest
    assert "zero-day" in manifest  # ressalvas metodológicas transportadas

    for name in (
        "figuras/convergencia_fc_dae_NSL-KDD.png",
        "figuras/curvas_roc_NSL-KDD.png",
        "figuras/recall_estratificado_NSL-KDD.png",
        "figuras/reconstrucao_diferencial_NSL-KDD.png",
        "figuras/ofat_L_NSL-KDD.png",
        "figuras/ablacao_deas_f1.png",
        "figuras/comparativo_f1.png",
        "tabelas/tabela4_NSL-KDD.tex",
        "tabelas/tabela4_NSL-KDD.md",
        "tabelas/ablacao_deas.tex",
        "tabelas/particionamento.tex",
    ):
        assert (out / name).exists(), name

    # Todo asset listado no manifesto existe em disco.
    for entry in payload["assets"]:
        assert (out / entry["path"]).exists(), entry["path"]


def test_missing_optional_artifacts_are_skipped_not_fatal(tmp_path):
    root = tmp_path / "reports"
    write_report(root, "NSL-KDD", "xAttack", "Normal", {"DoS": 30}, with_scores=False, with_ofat=False)
    payload = main(["--reports_dir", str(root), "--out_dir", str(tmp_path / "out")])

    reasons = " ".join(item["reason"] for item in payload["skipped"])
    assert "--save_scores" in reasons
    assert "--run_ofat" in reasons
    manifest = (tmp_path / "out" / "MANIFEST.md").read_text(encoding="utf-8")
    assert "Assets não gerados" in manifest


def test_stale_reports_are_refused(tmp_path):
    """Consolidar execuções de versões diferentes produziria tabelas incomparáveis."""
    root = tmp_path / "reports"
    write_report(root, "NSL-KDD", "xAttack", "Normal", {"DoS": 30})
    write_report(root, "ANTIGO", "Label", "BENIGN", {"DDoS": 30}, stale=True)

    with pytest.raises(ValueError, match="versão anterior da rotina"):
        discover_results(str(root))

    # A saída de emergência existe, e é explícita sobre o custo.
    results = discover_results(str(root), allow_stale=True)
    assert len(results) == 2


def test_subsampled_reports_emit_a_warning(tmp_path, capsys):
    root = tmp_path / "reports"
    write_report(root, "NSL-KDD", "xAttack", "Normal", {"DoS": 30}, subsampled=True)
    discover_results(str(root))
    assert "subamostrada" in capsys.readouterr().out


def test_fusion_degeneracy_is_detected_at_both_extremes():
    """alpha nos extremos colapsa B6a num dos termos e invalida a leitura de Delta_DEAS."""
    from src.evaluation.reporting import describe_fusion_degeneracy

    baselines = {
        "B2_Autoencoder_Alone": {"f1_score": 0.6706},
        "B3b_PreIF_DIF_Pure": {"f1_score": 0.4683},
        "B6a_Hybrid_Linear": {"f1_score": 0.6706},
    }

    collapsed_to_ae = describe_fusion_degeneracy(
        {"supervised": {"B6a_Hybrid_Linear_alpha": 0.0}}, baselines
    )
    assert collapsed_to_ae["equivalent_to"] == "B2_Autoencoder_Alone"
    assert collapsed_to_ae["metrics_identical"] is True
    assert "não" in collapsed_to_ae["message"]

    collapsed_to_dif = describe_fusion_degeneracy(
        {"supervised": {"B6a_Hybrid_Linear_alpha": 1.0}}, baselines
    )
    assert collapsed_to_dif["equivalent_to"] == "B3b_PreIF_DIF_Pure"

    # Uma fusão genuína não dispara o diagnóstico.
    assert describe_fusion_degeneracy(
        {"supervised": {"B6a_Hybrid_Linear_alpha": 0.72}}, baselines
    ) is None
    # Sem alfa calibrado não há o que diagnosticar.
    assert describe_fusion_degeneracy({}, baselines) is None


def test_manifest_warns_about_degenerate_fusion(tmp_path):
    root = tmp_path / "reports"
    directory = write_report(root, "CICIDS2017", "Label", "BENIGN", {"DDoS": 80})

    # Força o colapso: alpha = 0 e B6a idêntico a B2.
    path = directory / "experimental_results.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["calibration"]["supervised"]["B6a_Hybrid_Linear_alpha"] = 0.0
    payload["baselines"]["B6a_Hybrid_Linear"] = dict(payload["baselines"]["B2_Autoencoder_Alone"])
    path.write_text(json.dumps(payload), encoding="utf-8")

    out = tmp_path / "assets"
    main(["--reports_dir", str(root), "--out_dir", str(out)])

    manifest = (out / "MANIFEST.md").read_text(encoding="utf-8")
    assert "Fusão degenerada" in manifest
    assert "B2_Autoencoder_Alone" in manifest
    assert "não deve ser reportado como tal" in manifest


def test_consolidated_report_is_rebuilt_from_all_reports(tmp_path):
    """Rodar as bases em invocações separadas deixaria o consolidado incompleto."""
    root = tmp_path / "reports"
    write_report(root, "NSL-KDD", "xAttack", "Normal", {"DoS": 30})
    write_report(root, "UNSW-NB15", "attack_cat", "Normal", {"Exploits": 30})

    # Consolidado antigo, escrito por uma execução de um dataset só.
    (root / "consolidated_results.md").write_text("# antigo\nsó NSL-KDD\n", encoding="utf-8")

    payload = main(["--reports_dir", str(root), "--out_dir", str(tmp_path / "out")])

    consolidated = (root / "consolidated_results.md").read_text(encoding="utf-8")
    assert payload["consolidated"] is not None
    assert "antigo" not in consolidated
    assert "NSL-KDD" in consolidated and "UNSW-NB15" in consolidated


def test_filtered_run_does_not_overwrite_the_consolidated_report(tmp_path):
    """Com --datasets o conjunto é parcial: sobrescrever o consolidado o truncaria."""
    root = tmp_path / "reports"
    write_report(root, "NSL-KDD", "xAttack", "Normal", {"DoS": 30})
    write_report(root, "UNSW-NB15", "attack_cat", "Normal", {"Exploits": 30})
    (root / "consolidated_results.md").write_text("# completo\n", encoding="utf-8")

    payload = main(
        ["--reports_dir", str(root), "--out_dir", str(tmp_path / "out"), "--datasets", "NSL-KDD"]
    )

    assert payload["consolidated"] is None
    assert (root / "consolidated_results.md").read_text(encoding="utf-8") == "# completo\n"


def test_empty_reports_dir_is_rejected(tmp_path):
    (tmp_path / "vazio").mkdir()
    with pytest.raises(FileNotFoundError, match="Rode primeiro"):
        discover_results(str(tmp_path / "vazio"))
