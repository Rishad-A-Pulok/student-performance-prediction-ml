# Student Performance Prediction: Technical Report

## Abstract
Summarize research question, dataset, methods, and measured results **after execution**.

## 1. Introduction
Motivate timely identification of academic difficulty; do not claim validated interventions.

## 2. Data and target
395 mathematics records; G3 < 10 as positive at-risk class. Discuss sample and class counts from run_metadata.json.

## 3. Experimental setup
Describe fixed stratified split, CV model selection, preprocessing pipelines, scenarios, and metrics.

## 4. Results
Insert exact values from metrics.csv. Compare scenarios and inspect confusion matrices and ROC plots. Discuss false negatives and false positives.

## 5. Feature analysis
Discuss top features from permutation_importance.csv with uncertainty. Avoid causal interpretation.

## 6. Limitations and ethics
Address small sample, availability of features, domain shift to universities, demographic fairness, and lack of external validation.

## 7. Conclusion and future work
Conclude only what measured results support. Suggest repeated nested validation and prospective external evaluation.

## References
Cortez and Silva (2008); UCI Student Performance repository.
