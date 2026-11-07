# Phase A Baseline Extractor Evaluation Report

## 1. Baseline Summary
- **Evaluation Dataset**: 58 Gold Studies (696 Target Evaluations)
- **Overall Accuracy**: 78.45% (546 / 696)
- **Macro F1**: 0.7122
- **Total Errors**: 150 (115 False Positives / 35 False Negatives)
- **Exact Study Accuracy**: 8.62% (5 / 58)

## 2. Error Diagnostic Distribution
1. **Condition Conflation (103 errors, 68.7%)**: Extractor conflated mild degenerative changes (e.g. chondromalacia patellae) with osteoarthritis or tears.
2. **Lexicon Miss (32 errors, 21.3%)**: Extractor failed completely on non-English radiology reports (German, Spanish, Turkish, etc.).
3. **Compartment Leak (8 errors, 5.3%)**: Extractor assigned pathology to incorrect anatomical compartments.
4. **Negation / Polarity (7 errors, 4.7%)**: Extractor failed to detect complex linguistic negation syntax.
