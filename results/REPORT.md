# lesionSegmenter – report (2026-09-29)


Automatic outlining of EAE lesions from Cellpose segmentation masks and the collaborator's Pu.1 calls,
zoning into core / rim / peri-lesion, and selection of Pu.1⁺ cells per capture site for Deep Visual
Proteomics (laser microdissection + mass spectrometry). This report is generated from the run outputs;
the notebooks in `notebooks/` hold the image evidence behind every number.

**Decisions taken in this run** (all configurable in `configs/eae_dvp_sdata.yaml`):


- Input: SpatialData export `cellpose_output.zip` (4 scenes, 218,746 cells, label = cell_id; Pu.1 positivity from `Pu1_class`).
- Lesion score: **model** – a gradient-boosting model trained on the collaborator's manual CORE polygons, one leave-one-scene-out model per scene (each scene scored by a model that never saw its own annotations). Threshold {'type': 'absolute', 'value': 0.25, 'seed': 0.5}.
- Sections (animal + spinal level) from the curated `Sample_category` polygons; **only sections with manually annotated cores carry lesions** (`sections.focus: manual`); all other sections are lesion-free controls.
- Distances relative to section size: zone widths are fractions of each section's equivalent radius (rim 0.05, peri 0.15 → median 49 / 148 µm); rings for wells at 0–10 %, 10–20 %, 20–40 % of the radius; every cell carries `rel_pos` (0 = section centre / canal, 1 = pia).
- Manual annotations are used **only** for model training and validation, never to draw lesions directly.

## Headline numbers per scene

| scene | cells | Pu.1+ | Pu.1+ frac | lesions | lesion sections | sections | auto lesion mm² | manual core mm² | manual cores | cores ≥50% covered | manual area covered | wells |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CML_1 scene 0 | 57044 | 17315 | 0.30 | 22 | 5 | 9 | 1.95 | 0.69 | 34 | 0.71 | 0.90 | 40 |
| CML_2_rescan scene 0 | 60095 | 13214 | 0.22 | 30 | 3 | 9 | 1.78 | 0.97 | 25 | 0.92 | 0.85 | 30 |
| CML_metal scene 0 | 50627 | 15502 | 0.31 | 19 | 5 | 9 | 1.30 | 0.44 | 33 | 0.73 | 0.87 | 41 |
| CML_metal scene 1 | 50980 | 17992 | 0.35 | 30 | 3 | 9 | 2.23 | 0.36 | 20 | 0.95 | 0.97 | 29 |


## Pu.1⁺ cells from the segmentation mask (no zoning)

| sample | scene | n_cells | n_pu1_pos | frac_pu1 |
|---|---|---|---|---|
| CML_1 | 0 | 57044 | 17315 | 0.304 |
| CML_2_rescan | 0 | 60095 | 13214 | 0.220 |
| CML_metal | 0 | 50627 | 15502 | 0.306 |
| CML_metal | 1 | 50980 | 17992 | 0.353 |
| ALL |  | 218746 | 64023 | 0.293 |


Per animal and spinal level (T = thoracic, C = cervical, L = lumbar; CFA = adjuvant-only control, OS = other control):

| animal | level | n_sections | n_cells | n_pu1_pos | area_pu1_um2 | frac_pu1 |
|---|---|---|---|---|---|---|
| CFA_L2 | C | 2 | 9425 | 1397 | 48523.351 | 0.148 |
| CFA_L2 | L | 2 | 9944 | 1289 | 44694.480 | 0.130 |
| CFA_L2 | T | 2 | 4667 | 425 | 13682.922 | 0.091 |
| OS1_2 | nan | 2 | 5081 | 616 | 20062.261 | 0.121 |
| OS1_2 | C | 2 | 9248 | 1385 | 43038.029 | 0.150 |
| OS1_2 | L | 2 | 8107 | 1233 | 41263.947 | 0.152 |
| P2_3 | C | 2 | 19295 | 7355 | 210880.707 | 0.381 |
| P2_3 | L | 2 | 20138 | 6015 | 183980.948 | 0.299 |
| P2_3 | T | 2 | 14622 | 6734 | 181164.358 | 0.461 |
| P3_1 | C | 2 | 22451 | 7557 | 234566.699 | 0.337 |
| P3_1 | L | 2 | 8658 | 1646 | 49435.758 | 0.190 |
| P3_1 | T | 2 | 13290 | 3580 | 109248.933 | 0.269 |
| R1_2 | C | 2 | 1272 | 140 | 4404.661 | 0.110 |
| R1_2 | L | 2 | 11354 | 2757 | 79797.333 | 0.243 |
| R1_2 | T | 2 | 29670 | 12399 | 389588.871 | 0.418 |
| R1_3 | C | 2 | 11620 | 4044 | 108858.309 | 0.348 |
| R1_3 | L | 2 | 10362 | 3343 | 91108.823 | 0.323 |
| R1_3 | T | 2 | 8627 | 2058 | 57265.350 | 0.239 |
| unassigned | nan | 4 | 915 | 50 | 1455.538 | 0.055 |


## Sections

| scene | section_name | is_lesion_section | has_manual_core | area_mm2 | n_cells | n_pu1_pos | frac_pu1_pos | n_lesions | lesion_area_mm2 | lesion_frac_of_section | lesion_area_frac_unrestricted |
|---|---|---|---|---|---|---|---|---|---|---|---|
| CML_1 scene 0 | P2_3_T | True | True | 2.101 | 8293 | 3749 | 0.452 | 2 | 0.625 | 0.298 | 0.300 |
| CML_1 scene 0 | R1_3_C | False | False | 3.861 | 6085 | 2076 | 0.341 | 0 | 0.000 | 0.000 | 0.005 |
| CML_1 scene 0 | R1_3_L | True | True | 2.667 | 5523 | 1709 | 0.309 | 3 | 0.042 | 0.016 | 0.016 |
| CML_1 scene 0 | OS1_2_L | False | False | 3.026 | 4493 | 496 | 0.110 | 0 | 0.000 | 0.000 | 0.018 |
| CML_1 scene 0 | OS1_2_ | False | False | 2.390 | 2677 | 272 | 0.102 | 0 | 0.000 | 0.000 | 0.000 |
| CML_1 scene 0 | P2_3_C | True | True | 3.915 | 10078 | 4279 | 0.425 | 7 | 0.578 | 0.148 | 0.148 |
| CML_1 scene 0 | OS1_2_C | False | False | 4.241 | 4772 | 502 | 0.105 | 0 | 0.000 | 0.000 | 0.000 |
| CML_1 scene 0 | P2_3_L | True | True | 3.049 | 10589 | 3150 | 0.297 | 4 | 0.663 | 0.217 | 0.218 |
| CML_1 scene 0 | R1_3_T | True | True | 2.545 | 4475 | 1065 | 0.238 | 3 | 0.046 | 0.018 | 0.018 |
| CML_2_rescan scene 0 | R1_2_T | True | True | 5.456 | 15587 | 4735 | 0.304 | 12 | 0.941 | 0.172 | 0.172 |
| CML_2_rescan scene 0 | P3_1_C | True | True | 4.745 | 12370 | 3409 | 0.276 | 6 | 0.649 | 0.137 | 0.137 |
| CML_2_rescan scene 0 | P3_1_L | False | False | 2.794 | 4754 | 938 | 0.197 | 0 | 0.000 | 0.000 | 0.005 |
| CML_2_rescan scene 0 | CFA_L2_L | False | False | 4.591 | 5066 | 446 | 0.088 | 0 | 0.000 | 0.000 | 0.000 |
| CML_2_rescan scene 0 | CFA_L2_T | False | False | 2.567 | 2610 | 214 | 0.082 | 0 | 0.000 | 0.000 | 0.000 |
| CML_2_rescan scene 0 | R1_2_C | False | False | 0.922 | 896 | 121 | 0.135 | 0 | 0.000 | 0.000 | 0.000 |
| CML_2_rescan scene 0 | CFA_L2_C | False | False | 4.407 | 5401 | 468 | 0.087 | 0 | 0.000 | 0.000 | 0.000 |
| CML_2_rescan scene 0 | R1_2_L | False | False | 3.150 | 5878 | 1200 | 0.204 | 0 | 0.000 | 0.000 | 0.041 |
| CML_2_rescan scene 0 | P3_1_T | True | True | 4.236 | 7253 | 1680 | 0.232 | 4 | 0.188 | 0.044 | 0.044 |
| CML_metal scene 0 | P2_3_T | True | True | 1.979 | 6329 | 2985 | 0.472 | 2 | 0.510 | 0.258 | 0.258 |
| CML_metal scene 0 | R1_3_C | False | False | 4.014 | 5535 | 1968 | 0.356 | 0 | 0.000 | 0.000 | 0.002 |
| CML_metal scene 0 | R1_3_L | True | True | 2.775 | 4839 | 1634 | 0.338 | 3 | 0.028 | 0.010 | 0.010 |
| CML_metal scene 0 | OS1_2_L | False | False | 3.375 | 3614 | 737 | 0.204 | 0 | 0.000 | 0.000 | 0.006 |
| CML_metal scene 0 | OS1_2_ | False | False | 2.545 | 2404 | 344 | 0.143 | 0 | 0.000 | 0.000 | 0.000 |
| CML_metal scene 0 | P2_3_C | True | True | 4.238 | 9217 | 3076 | 0.334 | 4 | 0.395 | 0.093 | 0.093 |
| CML_metal scene 0 | OS1_2_C | False | False | 3.996 | 4476 | 883 | 0.197 | 0 | 0.000 | 0.000 | 0.000 |
| CML_metal scene 0 | P2_3_L | True | True | 3.089 | 9549 | 2865 | 0.300 | 6 | 0.353 | 0.114 | 0.114 |
| CML_metal scene 0 | R1_3_T | True | True | 2.577 | 4152 | 993 | 0.239 | 2 | 0.019 | 0.007 | 0.007 |
| CML_metal scene 1 | R1_2_T | True | True | 6.534 | 14083 | 7664 | 0.544 | 9 | 1.310 | 0.200 | 0.200 |
| CML_metal scene 1 | P3_1_C | True | True | 5.051 | 10081 | 4148 | 0.411 | 9 | 0.727 | 0.144 | 0.144 |
| CML_metal scene 1 | P3_1_L | False | False | 2.712 | 3904 | 708 | 0.181 | 0 | 0.000 | 0.000 | 0.002 |
| CML_metal scene 1 | CFA_L2_L | False | False | 5.834 | 4878 | 843 | 0.173 | 0 | 0.000 | 0.000 | 0.000 |
| CML_metal scene 1 | CFA_L2_T | False | False | 2.769 | 2057 | 211 | 0.103 | 0 | 0.000 | 0.000 | 0.000 |
| CML_metal scene 1 | R1_2_C | False | False | 0.936 | 376 | 19 | 0.051 | 0 | 0.000 | 0.000 | 0.000 |
| CML_metal scene 1 | CFA_L2_C | False | False | 4.006 | 4024 | 929 | 0.231 | 0 | 0.000 | 0.000 | 0.003 |
| CML_metal scene 1 | R1_2_L | False | False | 3.240 | 5476 | 1557 | 0.284 | 0 | 0.000 | 0.000 | 0.026 |
| CML_metal scene 1 | P3_1_T | True | True | 4.082 | 6037 | 1900 | 0.315 | 3 | 0.190 | 0.047 | 0.047 |


`lesion_area_frac_unrestricted` is what the detector found before control sections were cleared – the
sections R1_3_C, R1_2_L, P3_1_L (EAE animals without manual cores) still show Pu.1⁺ infiltrates there;
whether those are lesions is a decision for the reader (notebook 02, control-section panels).

## Validation against the manual cores

| scene | manual cores | cores ≥50% inside auto lesion | manual core area inside auto lesion | auto lesion mm² | manual core mm² |
|---|---|---|---|---|---|
| CML_1 scene 0 | 34 | 0.71 | 0.90 | 1.95 | 0.69 |
| CML_2_rescan scene 0 | 25 | 0.92 | 0.85 | 1.78 | 0.97 |
| CML_metal scene 0 | 33 | 0.73 | 0.87 | 1.30 | 0.44 |
| CML_metal scene 1 | 20 | 0.95 | 0.97 | 2.23 | 0.36 |


Model, leave-one-scene-out (bin level; avg_precision and cores_detected_frac are the informative columns):

| held_out | auc | avg_precision | n_cores | cores_detected_frac |
|---|---|---|---|---|
| CML_1_scene0 | 0.996 | 0.923 | 34 | 0.794 |
| CML_2_rescan_scene0 | 0.995 | 0.903 | 25 | 0.920 |
| CML_metal_scene0 | 0.996 | 0.839 | 33 | 0.818 |
| CML_metal_scene1 | 0.995 | 0.679 | 20 | 0.950 |


Threshold baseline (Pu.1⁺ density z-score) for comparison – best three of the grid:

| low | seed | min_area_um2 | det_frac | ctrl_area_mm2 | excess_frac | lesion_area_annot_mm2 | n_lesions | n_les_annot |
|---|---|---|---|---|---|---|---|---|
| 3.000 |  | 5000 | 0.955 | 0.140 | 0.380 | 10.430 | 190 | 138 |
| 2.500 |  | 5000 | 0.955 | 0.206 | 0.430 | 12.310 | 215 | 154 |
| 3.500 | 5.000 | 5000 | 0.946 | 0.087 | 0.320 | 9.000 | 149 | 111 |


The threshold detector kept ~94 % of manual cores but produced 3–5× the manual core area, including the central-canal region and grey matter; the model halves the excess and removes the canal calls at the price of missing some small manual cores (notebook 04 shows each missed core).


## Capture sites: Pu.1⁺ cells available for DVP

Pooled over all sections. `n_pu1_eligible` excludes cells within 100 µm of the section edge and inside VBO; `n_selected` are the cells placed in a 3000 µm² well.

| lesion_section | compartment | n_sections | n_cells_all | n_pu1_raw | n_pu1_eligible | area_pu1_eligible_um2 | n_selected | area_selected_um2 |
|---|---|---|---|---|---|---|---|---|
| False | GM | 10 | 27387 | 6168 | 6168 | 201764 | 1667 | 53041 |
| False | WM | 10 | 14004 | 1945 | 1875 | 53732 | 1022 | 27596 |
| True | core | 8 | 24935 | 11981 | 10939 | 312180 | 1310 | 36360 |
| True | rim | 8 | 28017 | 12013 | 9713 | 286756 | 1439 | 39677 |
| True | ring_0_10 | 8 | 23225 | 6525 | 5443 | 164818 | 1408 | 40304 |
| True | ring_10_20 | 8 | 14996 | 4351 | 3753 | 117601 | 1288 | 38462 |
| True | ring_20_40 | 8 | 23623 | 7539 | 6671 | 216282 | 1540 | 46983 |
| True | ring_40_60 | 8 | 15261 | 4370 | 3900 | 123504 | 1461 | 43427 |
| True | ring_60_plus | 8 | 8408 | 2276 | 1718 | 52073 | 699 | 19966 |


By manual annotation compartment (collaborator's GM / WM / VBO / core polygons):

| lesion_section | compartment | n_sections | n_cells_all | n_pu1_raw | n_pu1_eligible | area_pu1_eligible_um2 | n_selected | area_selected_um2 |
|---|---|---|---|---|---|---|---|---|
| False | manual:GM | 10 | 27387 | 6168 | 6168 | 201764 | 1667 | 53041 |
| False | manual:WM | 10 | 14004 | 1945 | 1875 | 53732 | 1022 | 27596 |
| False | manual:unannotated | 10 | 37480 | 6769 | 5115 | 151703 | 0 | 0 |
| False | manual:vbo | 10 | 505 | 50 | 0 | 0 | 0 | 0 |
| True | manual:core | 8 | 21907 | 10681 | 9410 | 275211 | 1539 | 42819 |
| True | manual:unannotated | 8 | 116145 | 38275 | 32716 | 997520 | 7589 | 221781 |
| True | manual:vbo | 8 | 403 | 85 | 0 | 0 | 0 | 0 |


Per section and per animal: `results/cohort/capture_sites_by_section.csv`, `capture_sites_by_animal.csv`; wells actually formed: `results/<sample>/scene<i>/wells.csv`.



## Mass-spec reaction plan (important-info.md)

Rules: ≤ 60 reactions of ~250 Pu.1⁺ cells; the two replicate slides of a section are pooled; lesion sections give one reaction per section per compartment (core, rim, peri, deep = remaining tissue outward of peri); control sections give GM and WM per section (CFA sections pooled together; other controls pooled only if the budget is exceeded); VBO gives one reaction per section that has VBO cells. Manual GM/WM/VBO polygons define those compartments; lesion compartments come from the automatic zones.

| item | value |
|---|---|
| reactions: GM | 5 |
| reactions: VBO | 15 |
| reactions: WM | 5 |
| reactions: lesion | 32 |
| reactions: total | 57 |
| max_reactions | 60 |
| within budget | True |
| target cells per reaction | 250 |
| reactions with shortfall | 20 |
| CFA GM/WM pooled | True |
| other control GM/WM pooled | True |
| control pooling level (0 none, 1 by prefix, 2 all) | 1 |
| reactions if other controls unpooled | 61 |
| reactions if controls pooled by prefix | 57 |
| reactions if other controls pooled | 51 |
| VBO sections | 15 |


**Plan** (`shortfall` = fewer than 80 % of the target available):

| reaction_id | reaction_name | pool_type | compartment | sections | n_sections | n_available | n_selected | shortfall |
|---|---|---|---|---|---|---|---|---|
| 1 | P2_3_C|core | lesion | core | P2_3_C | 1 | 1137 | 250 | False |
| 2 | P2_3_C|rim | lesion | rim | P2_3_C | 1 | 993 | 250 | False |
| 3 | P2_3_C|peri | lesion | peri | P2_3_C | 1 | 1191 | 250 | False |
| 4 | P2_3_C|deep | lesion | deep | P2_3_C | 1 | 1059 | 250 | False |
| 5 | P2_3_L|core | lesion | core | P2_3_L | 1 | 1726 | 250 | False |
| 6 | P2_3_L|rim | lesion | rim | P2_3_L | 1 | 501 | 250 | False |
| 7 | P2_3_L|peri | lesion | peri | P2_3_L | 1 | 736 | 250 | False |
| 8 | P2_3_L|deep | lesion | deep | P2_3_L | 1 | 681 | 250 | False |
| 9 | P2_3_T|core | lesion | core | P2_3_T | 1 | 2894 | 250 | False |
| 10 | P2_3_T|rim | lesion | rim | P2_3_T | 1 | 596 | 250 | False |
| 11 | P2_3_T|peri | lesion | peri | P2_3_T | 1 | 628 | 250 | False |
| 12 | P2_3_T|deep | lesion | deep | P2_3_T | 1 | 538 | 250 | False |
| 13 | P3_1_C|core | lesion | core | P3_1_C | 1 | 1712 | 250 | False |
| 14 | P3_1_C|rim | lesion | rim | P3_1_C | 1 | 2447 | 250 | False |
| 15 | P3_1_C|peri | lesion | peri | P3_1_C | 1 | 1054 | 250 | False |
| 16 | P3_1_C|deep | lesion | deep | P3_1_C | 1 | 915 | 250 | False |
| 17 | P3_1_T|core | lesion | core | P3_1_T | 1 | 564 | 250 | False |
| 18 | P3_1_T|rim | lesion | rim | P3_1_T | 1 | 708 | 250 | False |
| 19 | P3_1_T|peri | lesion | peri | P3_1_T | 1 | 543 | 250 | False |
| 20 | P3_1_T|deep | lesion | deep | P3_1_T | 1 | 350 | 250 | False |
| 21 | R1_2_T|core | lesion | core | R1_2_T | 1 | 2865 | 250 | False |
| 22 | R1_2_T|rim | lesion | rim | R1_2_T | 1 | 4240 | 250 | False |
| 23 | R1_2_T|peri | lesion | peri | R1_2_T | 1 | 2828 | 250 | False |
| 24 | R1_2_T|deep | lesion | deep | R1_2_T | 1 | 1393 | 250 | False |
| 25 | R1_3_L|core | lesion | core | R1_3_L | 1 | 28 | 28 | True |
| 26 | R1_3_L|rim | lesion | rim | R1_3_L | 1 | 143 | 143 | True |
| 27 | R1_3_L|peri | lesion | peri | R1_3_L | 1 | 223 | 223 | False |
| 28 | R1_3_L|deep | lesion | deep | R1_3_L | 1 | 294 | 250 | False |
| 29 | R1_3_T|core | lesion | core | R1_3_T | 1 | 13 | 13 | True |
| 30 | R1_3_T|rim | lesion | rim | R1_3_T | 1 | 85 | 85 | True |
| 31 | R1_3_T|peri | lesion | peri | R1_3_T | 1 | 116 | 116 | True |
| 32 | R1_3_T|deep | lesion | deep | R1_3_T | 1 | 202 | 202 | False |
| 33 | CFA(pooled)|GM | GM | GM | CFA_L2_C;CFA_L2_L;CFA_L2_T | 3 | 1626 | 250 | False |
| 34 | OS(pooled)|GM | GM | GM | OS1_2_;OS1_2_C;OS1_2_L | 3 | 1618 | 250 | False |
| 35 | P3_1_L|GM | GM | GM | P3_1_L | 1 | 437 | 250 | False |
| 36 | R1_2_L|GM | GM | GM | R1_2_L | 1 | 601 | 250 | False |
| 37 | R1_3_C|GM | GM | GM | R1_3_C | 1 | 1886 | 250 | False |
| 38 | CFA(pooled)|WM | WM | WM | CFA_L2_C;CFA_L2_L;CFA_L2_T | 3 | 221 | 221 | False |
| 39 | OS(pooled)|WM | WM | WM | OS1_2_;OS1_2_C;OS1_2_L | 3 | 304 | 250 | False |
| 40 | P3_1_L|WM | WM | WM | P3_1_L | 1 | 381 | 250 | False |
| 41 | R1_2_L|WM | WM | WM | R1_2_L | 1 | 547 | 250 | False |
| 42 | R1_3_C|WM | WM | WM | R1_3_C | 1 | 422 | 250 | False |
| 43 | CFA_L2_C|VBO | VBO | VBO | CFA_L2_C | 1 | 50 | 50 | True |
| 44 | CFA_L2_L|VBO | VBO | VBO | CFA_L2_L | 1 | 100 | 100 | True |
| 45 | CFA_L2_T|VBO | VBO | VBO | CFA_L2_T | 1 | 34 | 34 | True |
| 46 | OS1_2_C|VBO | VBO | VBO | OS1_2_C | 1 | 127 | 127 | True |
| 47 | OS1_2_L|VBO | VBO | VBO | OS1_2_L | 1 | 53 | 53 | True |
| 48 | P2_3_L|VBO | VBO | VBO | P2_3_L | 1 | 95 | 95 | True |
| 49 | P2_3_T|VBO | VBO | VBO | P2_3_T | 1 | 39 | 39 | True |
| 50 | P3_1_C|VBO | VBO | VBO | P3_1_C | 1 | 104 | 104 | True |
| 51 | P3_1_L|VBO | VBO | VBO | P3_1_L | 1 | 13 | 13 | True |
| 52 | P3_1_T|VBO | VBO | VBO | P3_1_T | 1 | 46 | 46 | True |
| 53 | R1_2_L|VBO | VBO | VBO | R1_2_L | 1 | 45 | 45 | True |
| 54 | R1_2_T|VBO | VBO | VBO | R1_2_T | 1 | 63 | 63 | True |
| 55 | R1_3_C|VBO | VBO | VBO | R1_3_C | 1 | 83 | 83 | True |
| 56 | R1_3_L|VBO | VBO | VBO | R1_3_L | 1 | 15 | 15 | True |
| 57 | R1_3_T|VBO | VBO | VBO | R1_3_T | 1 | 41 | 41 | True |


Alternative pooling of the non-CFA control GM/WM (the other setting of `pool_other_gm_wm`):

| item | value |
|---|---|
| reactions: GM | 7 |
| reactions: VBO | 15 |
| reactions: WM | 7 |
| reactions: lesion | 32 |
| reactions: total | 61 |
| max_reactions | 60 |
| within budget | False |
| target cells per reaction | 250 |
| reactions with shortfall | 23 |
| CFA GM/WM pooled | True |
| other control GM/WM pooled | False |
| control pooling level (0 none, 1 by prefix, 2 all) | 0 |
| reactions if other controls unpooled | 61 |
| reactions if controls pooled by prefix | 57 |
| reactions if other controls pooled | 51 |
| VBO sections | 15 |


Per-cell assignment for LMD: `results/<sample>/scene<i>/cells_reactions.csv` (`lesionseg export-lmd … --group-col reaction_name --reactions <that file>`).

## Figures

### CML_1 scene 0

![CML_1 scene 0](CML_1/scene0/overview.png)

Segmented Pu.1⁺ cells filled by zone, largest lesions:

![CML_1 scene 0 cells](CML_1/scene0/lesion_cells.png)

### CML_2_rescan scene 0

![CML_2_rescan scene 0](CML_2_rescan/scene0/overview.png)

Segmented Pu.1⁺ cells filled by zone, largest lesions:

![CML_2_rescan scene 0 cells](CML_2_rescan/scene0/lesion_cells.png)

### CML_metal scene 0

![CML_metal scene 0](CML_metal/scene0/overview.png)

Segmented Pu.1⁺ cells filled by zone, largest lesions:

![CML_metal scene 0 cells](CML_metal/scene0/lesion_cells.png)

### CML_metal scene 1

![CML_metal scene 1](CML_metal/scene1/overview.png)

Segmented Pu.1⁺ cells filled by zone, largest lesions:

![CML_metal scene 1 cells](CML_metal/scene1/lesion_cells.png)


## Notebooks (executed, in `notebooks/`)

1. `01_cohort_overview` – settings, headline numbers, Pu.1⁺ counts, overview figures, per-section bars
2. `02_judge_lesions` – every lesion section and every lesion at full resolution with automatic and manual outlines; control sections; the score map
3. `03_zones_relative_distance` – section radii, relative zone widths, Pu.1⁺ fraction vs relative distance, radial position of lesions, intensity by zone
4. `04_manual_vs_automatic` – model report, threshold baseline, missed manual cores and automatic lesions without a manual core, cell-level confusion
5. `05_capture_sites_and_wells` – capture-site tables (pooled, per section, per animal), wells, selected-cell maps, overlap with the collaborator's selection, LMD export demo

## Open decisions

- Accept the model-based outlines, or the threshold baseline, or tune the operating point (`lesion.threshold.value`, `.seed`)? Notebook 02/04 are the evidence.
- The EAE sections without manual cores (R1_3_C, R1_2_L, P3_1_L) show infiltrates – keep them lesion-free (current) or annotate them?
- Relative zone widths: rim 5 %, peri 15 % of section radius; rings 0–10 / 10–20 / 20–40 / 40–60 / >60 %. More material per well: raise `wells.target_area_um2`.
- LMD calibration marks are needed before `lesionseg export-lmd` can produce cut files.
- Scans 2023-1 and CML_2 have no Cellpose output yet.
