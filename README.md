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
   - Run the evaluation through `python ./eval/lceclassifier_seq_str.py`. Expected runing time: within 1 min Nvidia A100.
3. To reproduce the results on the third party dataset:
   - Run the evaluation through `python ./eval/lceclassifier_seq_str_bpags.py`. Expected runing time: within 1 min Nvidia A100.


### Expected output
The test results (Accuracy, Precision, Recall, F1, MCC) in the test set. The results will be outputed on the Terminal.



### Structure data
The structure data predicted via Alphafold 3 can be downloaded via: https://pan.baidu.com/share/init?surl=id_OhsVustY2lVSVsE52HQ&pwd=89as
