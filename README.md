# PLGDL
The code and datasets for the model PLGDL proposed in the paper: Leveraging Protein Language and Geometric Graph Models for Enhanced Vaccine Antigen Prediction

## To run the code:

### Requirements:

**Operating System**: Linux Ubuntu 20.04.

**Software dependencies and software versions**: please see `requirements.txt`

**Hardware**: CPU: Intel@ Xeon(R) Platinum 8360Y CPU @ 2.40GHzx 144, GPU: Nvidia A100



### Instructions
1. Install the required packages in requirement.py: `pip install -r requirements.txt`. **Normal install time**: within 1 hour.
2. To reproduce the results on our collected dataset:
   - Run the evaluation through `python a_final_code-lceclassifier_seq_str.py --seq_model_name self_seq --feature_type seq_strc --reduced_dim_strc 28 --reduced_dim_seq 255 --result_dir_suffix _change_strc_embed`. Expected runing time: within 1 min Nvidia A100.
3. To reproduce the results on the third party dataset:
   - Run the evaluation through `python a_final_code-lceclassifier_seq_str_bench.py --seq_model_name self_seq --feature_type seq_strc --reduced_dim_strc 28 --reduced_dim_seq 255 --result_dir_suffix _change_strc_embed`. Expected runing time: within 1 min Nvidia A100.


### Expected output
The test results (Accuracy, Precision, Recall, F1, MCC, AUC-ROC, AUC-PR) in the test set. The results will be outputed on the Terminal.
