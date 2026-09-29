**Calibração — CICIDS2017**

| Parâmetro | Valor | Regime |
| :--- | ---: | ---: |
| τ_AE (percentil p=95) | 0.000174 | Não supervisionado |
| τ_AE (μ + 3.0σ) | 0.004876 | Não supervisionado |
| τ_DIF_ref (percentil p=95) | 0.627946 | Não supervisionado |
| B1_iForest_Raw | 0.510173 | Supervisionado (Youden) |
| B2_Autoencoder_Alone | 0.000099 | Supervisionado (Youden) |
| B3_PreIF_iForest | 0.431712 | Supervisionado (Youden) |
| B3b_PreIF_DIF_Pure | 0.453374 | Supervisionado (Youden) |
| B6a_Hybrid_Linear_tau | 0.007529 | Supervisionado (Youden) |
| B6a_Hybrid_Linear_alpha | 0.000000 | Supervisionado (Youden) |
| B6b_Hybrid_OR_tau_dif | 0.503599 | Supervisionado (Youden) |
| B6b_Hybrid_OR_tau_ae | 0.000099 | Supervisionado (Youden) |

_Regra adotada para τ_AE: `percentile`. Os limiares não supervisionados usam apenas tráfego benigno de validação; os supervisionados usam a validação rotulada e nunca o teste._
