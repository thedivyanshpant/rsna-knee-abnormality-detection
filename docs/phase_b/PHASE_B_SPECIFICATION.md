# Phase B Labeling Pipeline Specification & Remediation History

## 1. Evolution of Phase B
1. **Phase B Step 1 (Assertion & Anatomical Scoping)**: 81.03% accuracy, 0.7397 F1 (18 errors fixed, 0 new errors, 7/7 polarity errors fixed).
2. **Phase B Step 2 (Clinical Severity Gating)**: 83.19% accuracy, 0.7623 F1 (15 errors fixed, condition conflation resolved).
3. **Phase B Step 3 Prototype (Multilingual Expansion)**: 84.77% accuracy, 0.7889 F1 (Achieved high Gold-58 score via aggressive string-stripping `re.sub` mutations).
4. **Phase B Step 3 Remediation (Non-Destructive Scoped & Soft Targets)**:
   - Eliminated destructive `re.sub` mutations.
   - Preserved exact character spans (`evidence_char_start`, `evidence_char_end`).
   - Introduced decoupled continuous soft targets $Y \in [0.02, 0.98]$, confidence $C$, and sample weights $W$.
   - Implemented active contradiction & epistemic hedge abstention ($W = 0.0$).
   - Verified 40/40 (100.00%) adversarial pass rate across 8 languages.
