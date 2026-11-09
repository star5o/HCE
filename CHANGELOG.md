# Changelog

## Discussion-period update (November 2026)

- **Loss defaults corrected.** `src/bdm/loss.py` and `scripts/train_bdm.py` previously defaulted to `M0 = 3.8`, `gamma = 1.0` and a smooth clamp. These defaults were not used for any reported result. Every reported BDM result was produced with `beta = 0.1`, `M0 = 3.0`, `gamma = 2.0`, a hard clamp on the raw margin, `-log sigmoid(beta * min(Delta, M_max))`, and the reference model's mean per-token entropy over the prompt tokens. The defaults now match this configuration, and the README formula is corrected accordingly (beta multiplies the clamped margin).
- **Original scripts added.** `scripts/original/` contains the training and evaluation scripts that produced the reported results, including the variants added during the discussion period (Eq. 6 as typeset, IPO, cDPO, hinge, and the NLL term).
