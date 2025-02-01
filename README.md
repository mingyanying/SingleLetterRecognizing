# SingleLetterRecognizing
the ANN one is ready.

# 01/17/2025 
loss of the ANN is 0.005 ish, and only for qwer this 4 letters, it cost too much time to train for the whole alphabet.

# 01/18/2025
loss of ANN is 0.005 ish, for all letters. 

# How the Filter is Learned?
During the forward pass, the filter extracts features from the input.
The backward pass computes the gradient of the loss with respect to the filter values.
The filter values are updated using gradient descent, making the filters adapt to the patterns in the training data.
