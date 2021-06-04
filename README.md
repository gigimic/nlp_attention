Attention Basics
----------------

Here we look at how attention is implemented. We will focus on implementing attention in isolation from a larger model. That's because when implementing attention in a real-world model, a lot of the focus goes into piping the data and juggling the various vectors rather than the concepts of attention themselves.

We will implement attention scoring and calculate an attention context vector.

We tried to calculate the annotation score with a single encoder hidden state and also 
with an annotation matrix. The dot() function of numpy was used here.
