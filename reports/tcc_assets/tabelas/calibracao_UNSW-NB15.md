**Calibração — UNSW-NB15**

| Parâmetro | Valor | Regime |
| :--- | ---: | ---: |
| τ_AE (percentil p=95) | 0.012036 | Não supervisionado |
| τ_AE (μ + 3.0σ) | 0.015563 | Não supervisionado |
| τ_DIF_ref (percentil p=95) | 0.634876 | Não supervisionado |
| B1_iForest_Raw | 0.423626 | Supervisionado (Youden) |
| B2_Autoencoder_Alone | 0.005369 | Supervisionado (Youden) |
| B3_PreIF_iForest | 0.497943 | Supervisionado (Youden) |
| B3b_PreIF_DIF_Pure | 0.476371 | Supervisionado (Youden) |
| B6a_Hybrid_Linear_tau | 0.479119 | Supervisionado (Youden) |
| B6a_Hybrid_Linear_alpha | 0.770000 | Supervisionado (Youden) |
| B6b_Hybrid_OR_tau_dif | 0.522446 | Supervisionado (Youden) |
| B6b_Hybrid_OR_tau_ae | 0.005369 | Supervisionado (Youden) |

_Regra adotada para τ_AE: `percentile`. Os limiares não supervisionados usam apenas tráfego benigno de validação; os supervisionados usam a validação rotulada e nunca o teste._
