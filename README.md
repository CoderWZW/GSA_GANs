this is the source file of paper: Gray-Box Shilling Attack An Adversarial Learning Approach. 


please visit link below to get QRec code that is to run recommendation algorithms, and put the codes into the file recommendation. 


operation step:
1. run getTargetsItem.py.
this step is to get target items that attempt to attack. If you succeed, you will see a file named "targets.txt"


2. run main.py.
this step is to generate data that based on comparison methods, including random attack, average attack, bandwagon attack and unorganized malicious attacks.


3. run GSA_GANs.py/ GSA_GANs_fixed.py.


GSA_GANs.py is for the recommendation model whose parameters are changable.
GSA_GANs.py is for the recommendation model whose parameters are fixed.


4. run prediction/evaluation.py


this step is to calculate the prediction shift/hit ratio.
