# score-emotion-classifier
Code accompanying an MSc dissertation on emotion classification in student counselling self-referral text.

## Project Purpose
- The code in this repository was used for an MSc Psychology project in which the aim was to develop a sentence-transformer classifier that could reliably identify emotion within text based on limited data and CPU capabilities. The emotion labels provided by the optimal model and training configuration were then used in a logistic regression to test whether each emotion's presence could be predictive on non-attendance at a first offered counselling appointment.

## Data Governance
- The raw data that was used for both the model development/evaluation and logistic regression phases cannot be shared due to confidentiality. The raw data remained on a secure drive within university servers at all times and thus cannot be shared anywhere.

## Folders
- The python folder contains all python scripts used during the project, used for a variety of purposes detailed below.
- The R folder contains all R scripts used during the logistic regression phase of the project.
- The environments folder contains two files, each a snapshot of the two environments (classy and setfit) used for the two model training approaches.

## Python Script Order
- 1) build_template.py - constructs an excel file to act as a template for annotating/labelling each sentence unit. Uses the raw data file and splits each document into constituent sentences by the presence of select punctuation marks and a space. The template contains a column for document ID, the content of the individual sentence and labels being applied to that sentence.
  2) prepare_sentences.py - takes the template file once annotations have been applied and converts the annotations into a binary matrix (sentences down the side, presence of classes (0 or 1) along the top). This format is required for the training approaches.
  3) make_splits.py - uses the output from prepare_sentences.py and splits the data into five stratified folds for development/evaluation. Four folds are assigned for development and one reserved for evaluation, with positive examples of classes being distributed evenly and in a way that keeps all sentences of the same document together.
  4) cc_evaluate.py/setfit_evaluate.py - these are the training files for the two training approaches, calling on either Classy Classification or SetFit to train a selected backbone model using the matrix produced by prepare_sentences.py and the splits determined by make_splits.py. Each uses k-fold validation method to train and produce predictions for all sentences as if they had not been seen before, and a one-vs-rest strategy for classification. These files were used during development in order to gauge the performance of the classifiers, and informed the decision to target-sample for the rarer classes.
  5) add_target_sampled_folds.py - adds the sentences from the target sampled documents to the four training folds made by make_splits.py, without adding any to the evaluation fold. Requires the sentences to be added to the template excel file, and for prepare_sentences.py to have been run again before hand.
  6) select_thresholds.py - selects optimum per-class thresholds based on which probability threshold is associated with the best F1 score for the class. These thresholds are then used for final evaluation.
  7) evaluate_test_set.py - trains the backbone model on all data from the training folds and then provides predictions for the reserved evaluation fold based on the per-class thresholds selected by select_thresholds.py.
  8) build_regression_dataset.py - takes information from several files/outputs across the project to build a dataset excel file to use for the logistic regression side of the project. Information it gathers is: demographic and control data from a file shared on the secure drive (unblinded after evaluate_test_set.py was run), hand-annotated labels from the template file, out-of-fold predictions for the training fold documents from the output of cc/setfit_evaluate.py and predictions for the evaluation fold documents from the output of evaluate_test_set.py. The output is a dataset ready to be used for statistical analysis.
