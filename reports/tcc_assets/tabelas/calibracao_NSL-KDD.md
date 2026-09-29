**Calibração — NSL-KDD**

| Parâmetro | Valor | Regime |
| :--- | ---: | ---: |
| τ_AE (percentil p=95) | 0.023966 | Não supervisionado |
| τ_AE (μ + 3.0σ) | 0.031035 | Não supervisionado |
| τ_DIF_ref (percentil p=95) | 0.649353 | Não supervisionado |
| B1_iForest_Raw | 0.484057 | Supervisionado (Youden) |
| B2_Autoencoder_Alone | 0.012295 | Supervisionado (Youden) |
| B3_PreIF_iForest | 0.523446 | Supervisionado (Youden) |
| B3b_PreIF_DIF_Pure | 0.515713 | Supervisionado (Youden) |
| B6a_Hybrid_Linear_tau | 0.508990 | Supervisionado (Youden) |
| B6a_Hybrid_Linear_alpha | 0.720000 | Supervisionado (Youden) |
| B6b_Hybrid_OR_tau_dif | 0.562155 | Supervisionado (Youden) |
| B6b_Hybrid_OR_tau_ae | 0.012295 | Supervisionado (Youden) |

_Regra adotada para τ_AE: `percentile`. Os limiares não supervisionados usam apenas tráfego benigno de validação; os supervisionados usam a validação rotulada e nunca o teste._
