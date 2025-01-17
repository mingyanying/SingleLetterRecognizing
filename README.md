# SingleLetterRecognizing
Just like MNIST, but with alphbets.

# 01/17/2025 
loss of the ANN is 0.005 ish, and only for qwer this 4 letters, it cost too much time to train for the whole alphabet, ganna try CNN next.

# How the Filter is Learned (Anyone can explain why the gradient thingy here...)
During the forward pass, the filter extracts features from the input.
The backward pass computes the gradient of the loss with respect to the filter values.
The filter values are updated using gradient descent, making the filters adapt to the patterns in the training data.