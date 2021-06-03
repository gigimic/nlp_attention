import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

dec_hidden_state = [5, 1, 20]

'''We will assume we're in the first step in the decoding phase. 
The first input to the scoring function is the hidden state of decoder 
(assuming a toy RNN with three hidden nodes -- only to  illustrate)'''

# visualize decoder hidden state
plt.figure(figsize=(1.5, 4.5))
sns.heatmap(np.transpose(np.matrix(dec_hidden_state)), annot=True, cmap=sns.light_palette("purple", as_cmap=True), linewidths=1)
plt.show()