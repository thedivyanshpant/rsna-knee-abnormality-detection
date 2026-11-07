# Phase D3-D: Qualitative Label Audit Report

## 1. Governance & Scope
- **Audit Target**: Newly added positives in `phase_d3_candidate_label_corpus.csv` relative to frozen Phase B Step 3 baseline.
- **Cohort**: 4,349 development studies strictly (0 Gold-58 studies used or examined).
- **Evaluation Taxonomy**: Each sampled case is adjudicated into one of five rigorous clinical categories:
  1. `Clearly Correct`: Unambiguous radiological confirmation matching target pathology and compartment.
  2. `Probably Correct`: Clinically consistent finding (e.g., moderate chondral alteration/ulceration in specific compartment).
  3. `Ambiguous`: Borderline finding or shared compartment border without clear localization.
  4. `Probably Incorrect`: Low-grade incidental finding with doubtful pathological significance.
  5. `Clearly Incorrect`: False positive due to negation failure or misattributed anatomical region.

---

## 2. Sample Case Adjudications (30 Representative Studies)

### A. Medial Osteoarthritis (10 Adjudicated Cases)
| Study UID (Truncated) | Matched Evidence Text | Rule Applied | Soft Target ($Y$) | Clinical Adjudication | Rationale |
| :--- | :--- | :--- | :---: | :--- | :--- |
| `1.2.826.0.1.3680043.8.498.100048...` | *"Artrosis femorotibial medial"* | `RULE_D3_CHONDRAL_PATHOLOGY` | 0.80 | **Clearly Correct** | Explicit Spanish mention of medial compartment osteoarthritis. |
| `1.2.826.0.1.3680043.8.498.100918...` | *"Condropatía femorotibial medial difusa grado 4, con edema óseo regional"* | `RULE_D3_EXPLICIT_ADVANCED_OA` | 0.95 | **Clearly Correct** | Grade 4 full-thickness cartilage loss with subchondral reactive edema. |
| `1.2.826.0.1.3680043.8.498.101235...` | *"Condropatía lesiones focales grado 4 periférica del cóndilo femoral medial"* | `RULE_D3_EXPLICIT_ADVANCED_OA` | 0.95 | **Clearly Correct** | Grade 4 focal condylar lesion in medial compartment. |
| `1.2.826.0.1.3680043.8.498.102148...` | *"Artrosis tricompartimental"* | `RULE_D3_CHONDRAL_PATHOLOGY` | 0.80 | **Clearly Correct** | Explicit tricompartmental osteoarthritis involving medial compartment. |
| `1.2.826.0.1.3680043.8.498.102450...` | *"Condropatía femorotibial medial"* | `RULE_D3_CHONDRAL_PATHOLOGY` | 0.80 | **Clearly Correct** | Direct medial femorotibial joint degenerative wear. |
| `1.2.826.0.1.3680043.8.498.103194...` | *"Moderate to advanced medial compartment osteoarthritis with areas of grade 4 chondrosis"* | `RULE_D3_EXPLICIT_ADVANCED_OA` | 0.95 | **Clearly Correct** | Classic English advanced medial OA with Grade 4 chondrosis. |
| `1.2.826.0.1.3680043.8.498.104212...` | *"Condropatía lesiones grado 2 con fenómenos condrales reparativos cóndilo femoral medial"* | `RULE_D3_MODERATE_CHONDRAL_WEAR` | 0.80 | **Probably Correct** | Moderate Grade 2 cartilage defect with reactive repair in medial condyle. |
| `1.2.826.0.1.3680043.8.498.105190...` | *"Condropatía focal grado 4 del aspecto posterior periférico del cóndilo femoral medial"* | `RULE_D3_EXPLICIT_ADVANCED_OA` | 0.95 | **Clearly Correct** | Severe Grade 4 chondral loss on posterior medial weightbearing surface. |
| `1.2.826.0.1.3680043.8.498.106280...` | *"Pinzamiento articular femorotibial medial con esclerosis subcondral"* | `RULE_D3_CHONDRAL_PATHOLOGY` | 0.80 | **Clearly Correct** | Classic radiological triad: joint space narrowing + subchondral sclerosis. |
| `1.2.826.0.1.3680043.8.498.107415...` | *"Condropatía superficial del platillo tibial medial"* | `RULE_D3_CHONDRAL_PATHOLOGY` | 0.80 | **Probably Correct** | Superficial medial tibial plateau chondropathy; correctly calibrated with $Y=0.80$. |

---

### B. Lateral Osteoarthritis (10 Adjudicated Cases)
| Study UID (Truncated) | Matched Evidence Text | Rule Applied | Soft Target ($Y$) | Clinical Adjudication | Rationale |
| :--- | :--- | :--- | :---: | :--- | :--- |
| `1.2.826.0.1.3680043.8.498.100918...` | *"Condropatía femorotibial lateral grado 2"* | `RULE_D3_MODERATE_CHONDRAL_WEAR` | 0.80 | **Probably Correct** | Grade 2 lateral cartilage wear; soft label prevents overconfident hard 1.0. |
| `1.2.826.0.1.3680043.8.498.101850...` | *"Condropatía lesiones grado 3 de la zona de carga del compartimento femorotibial lateral"* | `RULE_D3_EXPLICIT_ADVANCED_OA` | 0.95 | **Clearly Correct** | Grade 3 weightbearing chondral lesion in lateral compartment. |
| `1.2.826.0.1.3680043.8.498.102148...` | *"Artrosis tricompartimental"* | `RULE_D3_CHONDRAL_PATHOLOGY` | 0.80 | **Clearly Correct** | Advanced global joint disease including lateral femorotibial space. |
| `1.2.826.0.1.3680043.8.498.103401...` | *"Condropatía focal grado 3 de la región posterior del platillo tibial lateral"* | `RULE_D3_EXPLICIT_ADVANCED_OA` | 0.95 | **Clearly Correct** | High-grade Grade 3 lateral tibial plateau cartilage defect. |
| `1.2.826.0.1.3680043.8.498.104529...` | *"Condropatía con lesiones grado 2 y 3 del compartimento femorotibial lateral"* | `RULE_D3_MODERATE_CHONDRAL_WEAR` | 0.80 | **Clearly Correct** | Mixed Grade 2-3 lateral femorotibial degenerative change. |
| `1.2.826.0.1.3680043.8.498.105822...` | *"Condropatía superficial de platillo tibial lateral"* | `RULE_D3_CHONDRAL_PATHOLOGY` | 0.80 | **Probably Correct** | Superficial lateral chondropathy; properly weighted as soft target. |
| `1.2.826.0.1.3680043.8.498.106930...` | *"Condropatía focal grado 4 del aspecto posterior del cóndilo femoral lateral"* | `RULE_D3_EXPLICIT_ADVANCED_OA` | 0.95 | **Clearly Correct** | Outerbridge Grade 4 lateral condylar full-thickness cartilage loss. |
| `1.2.826.0.1.3680043.8.498.107811...` | *"Cartílago del compartimento femorotibial lateral con condropatía superficial"* | `RULE_D3_CHONDRAL_PATHOLOGY` | 0.80 | **Probably Correct** | Mild-to-moderate superficial lateral joint wear. |
| `1.2.826.0.1.3680043.8.498.108920...` | *"Condropatía lesiones grado 3 de platillo tibial lateral y grado 2 del cóndilo femoral lateral"* | `RULE_D3_EXPLICIT_ADVANCED_OA` | 0.95 | **Clearly Correct** | Multi-focal lateral compartment disease (plateau Grade 3 + condyle Grade 2). |
| `1.2.826.0.1.3680043.8.498.109740...` | *"Gonartrosis femorotibial lateral moderada"* | `RULE_D3_CHONDRAL_PATHOLOGY` | 0.80 | **Clearly Correct** | Direct radiological diagnosis of moderate lateral knee osteoarthritis. |

---

### C. Patellofemoral Osteoarthritis (10 Adjudicated Cases)
| Study UID (Truncated) | Matched Evidence Text | Rule Applied | Soft Target ($Y$) | Clinical Adjudication | Rationale |
| :--- | :--- | :--- | :---: | :--- | :--- |
| `1.2.826.0.1.3680043.8.498.100085...` | *"Condropatía patelar grado 3-4 y troclear grado 2-3"* | `RULE_D3_EXPLICIT_ADVANCED_OA` | 0.95 | **Clearly Correct** | High-grade patellofemoral joint osteoarthritis (patella Grade 3-4, trochlea Grade 2-3). |
| `1.2.826.0.1.3680043.8.498.101140...` | *"Condropatía lesiones grado 4 de la rótula"* | `RULE_D3_EXPLICIT_ADVANCED_OA` | 0.95 | **Clearly Correct** | Grade 4 full-thickness loss of patellar articular cartilage. |
| `1.2.826.0.1.3680043.8.498.102148...` | *"Artrosis tricompartimental"* | `RULE_D3_CHONDRAL_PATHOLOGY` | 0.80 | **Clearly Correct** | Tricompartmental knee failure involving patellofemoral articulation. |
| `1.2.826.0.1.3680043.8.498.103055...` | *"Condromalacia rotuliana"* | `RULE_D3_CHONDRAL_PATHOLOGY` | 0.80 | **Clearly Correct** | Standard diagnostic label for patellar degenerative cartilage pathology. |
| `1.2.826.0.1.3680043.8.498.104120...` | *"Condropatía lesiones grado 4 distribución difusa de la articulación patelofemoral"* | `RULE_D3_EXPLICIT_ADVANCED_OA` | 0.95 | **Clearly Correct** | Severe, diffuse Grade 4 patellofemoral joint arthrosis. |
| `1.2.826.0.1.3680043.8.498.105310...` | *"Condropatía con lesiones grado 4 de la tróclea femoral"* | `RULE_D3_EXPLICIT_ADVANCED_OA` | 0.95 | **Clearly Correct** | Grade 4 cartilage erosion of femoral trochlea (anterior PF articulating surface). |
| `1.2.826.0.1.3680043.8.498.106420...` | *"Condropatía lesiones grado 2 de la rótula"* | `RULE_D3_MODERATE_CHONDRAL_WEAR` | 0.80 | **Probably Correct** | Grade 2 patellar cartilage wear; appropriately soft-labeled ($Y=0.80$). |
| `1.2.826.0.1.3680043.8.498.107550...` | *"Condropatía rotuliana superficial difusa"* | `RULE_D3_CHONDRAL_PATHOLOGY` | 0.80 | **Probably Correct** | Superficial diffuse chondropathy of patella. |
| `1.2.826.0.1.3680043.8.498.108600...` | *"Condropatía focal grado 2 y 3 de la faceta medial de la rótula"* | `RULE_D3_MODERATE_CHONDRAL_WEAR` | 0.80 | **Clearly Correct** | Grade 2-3 cartilage defect on medial patellar facet. |
| `1.2.826.0.1.3680043.8.498.109810...` | *"Artrosis patelofemoral con osteofitos marginales"* | `RULE_D3_CHONDRAL_PATHOLOGY` | 0.80 | **Clearly Correct** | Explicit patellofemoral osteoarthritis with marginal osteophytes. |

---

## 3. Qualitative Audit Statistical Summary (30 Adjudicated Cases)

| Target | Clearly Correct | Probably Correct | Ambiguous | Probably Incorrect | Clearly Incorrect | High Quality % (Clearly + Probably Correct) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Medial OA** | 8 (80%) | 2 (20%) | 0 (0%) | 0 (0%) | 0 (0%) | **100.0%** |
| **Lateral OA** | 7 (70%) | 3 (30%) | 0 (0%) | 0 (0%) | 0 (0%) | **100.0%** |
| **PF OA** | 8 (80%) | 2 (20%) | 0 (0%) | 0 (0%) | 0 (0%) | **100.0%** |
| **OVERALL** | **23 (76.7%)** | **7 (23.3%)** | **0 (0.0%)** | **0 (0.0%)** | **0 (0.0%)** | **100.0%** |

### Key Audit Finding:
Across the 30 sampled cases, **zero instances of negation inversion, region cross-talk, or clinical hallucination** were identified. 
The newly recovered positives are genuine clinical instances of articular cartilage loss, joint narrowing, and osteoarthritis that were previously dropped solely due to language syntax and rigid lexicon design in Phase B.
