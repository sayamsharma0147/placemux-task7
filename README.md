Task 7 — Baseline Target Feature Engineering
PlaceMux · Altrodav Technologies · Phase 1 Industry Immersion
Objective
Engineer informative features from raw question-bank data, check every feature for leakage, rank by importance, prune low-value ones, and lock a vetted baseline feature set.
Dataset
6 question-bank xlsx files → 4,799 rows across 6 domains (AI & ML, Logical Reasoning, Numerical Ability, React Engineer, SAP Engineer, Verbal Ability). DevOps excluded — malformed header.
Target
difficulty_level binarized at median (42) → Easy=0, Hard=1. Near-balanced: 48.6% / 51.4%.
Features Engineered (11 total)
FeatureDescriptionGroupq_lenQuestion character lengthTextq_word_countQuestion word countTextavg_opt_lenMean length of 4 optionsOptionsmax_opt_lenLongest option lengthOptionsopt_len_rangeMax – Min option lengthOptionstotal_opt_lenSum of all option lengthsOptionsq_to_avg_opt_ratioQuestion length / avg option lengthRatiotopic_avg_difficultyMean difficulty per topic (train-only)Aggregatedomain_avg_difficultyMean difficulty per domain (train-only)Aggregatetopic_encTopic label encodedCategoricaldomain_encDomain label encodedCategorical
Leakage Check
FeatureStatusActionText/option length features✅ SafeUsed directlydomain_enc, topic_enc✅ SafeUsed directlydomain_avg_difficulty⚠️ FixedRecomputed from train-onlytopic_avg_difficulty⚠️ FixedRecomputed from train-onlydifficulty_level, question_id, correct_answer❌ ExcludedDirect leakage / no signal
Results
StageAccuracyF1 (Hard)All features (val)0.6360.648Pruned baseline (val)0.6440.658Locked baseline (test)0.6470.649
Prune threshold: 0.015 importance. No features dropped — all 11 cleared the bar.
Files
FileDescriptiontask7_feature_engineering.pyFull pipeline with detailed commentstask7_feature_engineering.pngImportance chart + F1 comparisontask7_experiment_log.jsonStructured run log with all decisions
How to Run
bashpip install scikit-learn matplotlib openpyxl pandas
python task7_feature_engineering.py
