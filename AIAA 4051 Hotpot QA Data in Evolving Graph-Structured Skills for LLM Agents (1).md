Construct reproducible train, validation, and test splits from the original **HotpotQA dataset under the distractor setting**, using a random seed of your choice. Use the dataset’s original difficulty labels (`easy`, `medium`, and `hard`) for stratified sampling.

| Project split | Source file (full URL)                                       | Sampling requirement                                         | Total |
| ------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ----- |
| Train         | http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json | Randomly sample 100 questions from each difficulty level     | 300   |
| Validation    | http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json | Randomly sample 50 questions from each difficulty level, excluding all selected training questions | 150   |
| Test          | http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json | Randomly sample 200 questions                                | 1000  |

Sample without replacement. The training and validation sets must contain **no overlapping questions**. Verify that all three project splits are mutually disjoint using their original question IDs, and use the same fixed splits across all experimental conditions.

Keep the supplied-context setting fixed: provide each question together with all context paragraphs included in the distractor dataset. Do not add external retrieval or web search, and do not include reference answers or supporting-fact labels in the solver input.

For reproducibility, **report the selected random seed, dataset version, and sampling procedure in the experimental report**. Include the exact original question IDs in the submitted code, organized into three lists: `train_ids`, `val_ids`, and `test_ids`. These IDs must correspond to HotpotQA’s original `_id` field, or the equivalent `id` field in its Hugging Face representation, so that split membership, sample counts, and non-overlap can be independently verified.

Use training data for skill optimization and validation data for hyperparameter tuning and skill selection. Freeze the selected skill and inference settings before evaluating on the test set. Clearly state in the report that the project’s test set consists of **200 questions sampled from the official distractor development set**, rather than the official hidden test set.