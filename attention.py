import numpy as np
import matplotlib.pyplot as plt
from numpy.core.numeric import _ones_like_dispatcher
import seaborn as sns

dec_hidden_state = [5, 1, 20]

'''We will assume we're in the first step in the decoding phase. 
The first input to the scoring function is the hidden state of decoder 
(assuming a toy RNN with three hidden nodes -- only to  illustrate)'''

# visualize decoder hidden state
# plt.figure(figsize=(1.5, 4.5))
# sns.heatmap(np.transpose(np.matrix(dec_hidden_state)), annot=True, cmap=sns.light_palette("purple", as_cmap=True), linewidths=1)
# plt.show()

# the first scoring function will score a single annotation (encoder hidden state), 
annotation = [3,12,45] #e.g. Encoder hidden state


# we can visualize the single annotation
# plt.figure(figsize=(1.5, 4.5))
# sns.heatmap(np.transpose(np.matrix(annotation)), annot=True, cmap=sns.light_palette("orange", as_cmap=True), linewidths=1)
# plt.show()

# use numpy's dot() to calculate the dot product of a single annotation. 
def single_dot_attention_score(dec_hidden_state, enc_hidden_state):
    # return the dot product of the two vectors
    return np.dot(dec_hidden_state, enc_hidden_state)
    

dot_score = single_dot_attention_score(dec_hidden_state, annotation)
print('dot product of the sigle annotation is   ', dot_score)

# we can calculate the scoring all the annotations at once using the following annotation matrix.
annotations = np.transpose([[3,12,45], [59,2,5], [1,43,5], [4,3,45.3]])
# we can visualize our annotation (each column is an annotation)
 
# Now we need to score all annotations at _once 
def dot_attention_score(dec_hidden_state, annotations):
    #  return the product of dec_hidden_state transpose and enc_hidden_states
    return np.matmul(np.transpose(dec_hidden_state), annotations)
    
attention_weights_raw = dot_attention_score(dec_hidden_state, annotations)
print('attention score matrix  ', attention_weights_raw)
