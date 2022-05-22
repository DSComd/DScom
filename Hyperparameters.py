'''
==========================================
        [Hyperparameters]
==========================================
'''

"""
___________________________________

For Diffusion Chain Generation
___________________________________
"""

Diff_Model = "General_IC"
# should be one of the following.
# "IC", "LT", "General_IC"

Num_Chains = 500
# the least number of samples we want to generate

Window_length_half = 2   
# We want nodes on the same diffusion chain within the window to be 
#       more similar to each other.

v = None
Feature_concat = 6
W = None
# v and W are the parameters of out IC model.
# Feature_concat is the length of v_a.
# if v and W are None, we will randomly generate v and W.

# Specially, we use the first element in node feature.

Scalar_factor = 0.5
Offset = -6
# In our IC model, p(u->v) = sigmoid(a*score(h_t,h_s)+b), where
#       a is the Scalar_factor here, b is the Offset here.

Random_Seed = 20220519
# For random number generation.


"""
___________________________________

For DSCOM - attention Model
___________________________________
"""

Learning_rate = 0.01
# Learning Rate.

Training_epoch = 1000
# The number of epoches in the training period.

Out_num_features = 6
# The number of the features in the embedding of nodes after the
#     neural network.

Negative_sample_ratio = 5
# The ratio of #(negative samples) / #(positive samples).

# Num_Seeds = 20 
# NOTE: Now Num_Seeds should be assigned in "Evaluation-formal.ipynb".
# The number of seeds we want to select on the dataset.
