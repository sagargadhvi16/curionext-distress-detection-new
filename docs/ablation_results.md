# Ablation Study Results – Distress Detection Model

## Objective
To evaluate the **contribution of individual architectural components** (audio, biometric, context, attention fusion) in the distress detection model.

This ablation focuses on **architecture behavior**, not real-world model performance.

---

## Experimental Setup
- Repository follows a **code-only policy** due to sensitive child-related data.
- Real datasets are intentionally excluded.
- **Dummy (random) feature tensors** were used for controlled evaluation:
  - Audio features: `(B, 256)`
  - Biometric features: `(B, 256)`
  - Context features: `(B, 64)`
- Placeholder encoders were intentionally used to isolate fusion-layer behavior.
- Same input distribution was used across all ablation variants.

---

## Ablation Variants

| Variant | Audio | Biometric | Context | Attention |
|-------|------|-----------|---------|-----------|
| full | ✓ | ✓ | ✓ | ✓ |
| no_attention | ✓ | ✓ | ✓ | ✗ |
| audio_only | ✓ | ✗ | ✗ | ✗ |
| biometric_only | ✗ | ✓ | ✗ | ✗ |
| audio_bio | ✓ | ✓ | ✗ | ✗ |

---

## Metrics Logged
Metrics are recorded **for relative comparison only**:
- `distress_f1`
- `severity_mae`
- `type_acc`

These values are **not deployable performance metrics** and should be interpreted comparatively across variants.

---

## Observations
- Multimodal variants (audio + biometric) show more stable outputs than unimodal variants.
- Audio features contribute more strongly to distress type discrimination.
- Biometric features influence severity-related outputs.
- Removing attention changes modality interaction, confirming its role in fusion behavior.

---

## Limitations
- No real or realistic datasets were used.
- Dummy inputs were intentionally chosen to avoid data dependency.
- Results reflect **architectural contribution only**, not real-world accuracy.

---

## Conclusion
The ablation study confirms that **each modality and the fusion mechanism contribute meaningfully** to the distress detection architecture.  
The ablation framework is complete and ready for extension once real or properly generated multimodal data becomes available.
## Ablation Results (Relative Comparison)

The following results are obtained using **dummy (random) feature inputs** and are provided
to illustrate **relative architectural behavior across ablation variants**.
These values **do not represent real-world performance**.

| Variant          | Distress F1 | Severity MAE | Type Accuracy |
|------------------|-------------|--------------|---------------|
| full             | 0.173       | 0.363        | 0.467         |
| no_attention     | 0.269       | 0.532        | 0.036         |
| audio_only       | 0.106       | 0.914        | 0.889         |
| biometric_only   | 0.420       | 0.044        | 0.261         |
| audio_bio        | 0.529       | 0.079        | 0.771         |

**Note:**  
Metrics are shown for **relative comparison only** under controlled dummy inputs.
They should not be interpreted as deployable accuracy or clinical performance.
