
Training LLM from Scratch

Recently was working on write the transformers architecture code from scratch and training it on GPU's. The architecture used is different from the vanilla Attention is all you need , Vaswani et al. 2017. I used the Rotary positional encoding for query and keys, use rms norm as pre-norm and SwigluFFN for feed forward layers.

I trained the LLM on A-100 GPU multiple times with different hyperparameters. The LLM was trained on TinyStories dataset on 2M tokens. (Same dataset was used to build vocab for tokenizers)
The model which have lowest loss in training:
Transformer_LM(vocab_size=3000, context_length=300, d_model=32, num_layers=8, num_heads=4, d_ff=128)
 Loss -> 5.88
![[Pasted image 20250920204043.png]]

Below is github link for the code:
scratch.py file has components written in scratch for
Linear layer, Softmax, SwigluFFN, Rope, Embedding model, RMS Norm, Multihead attention

training_llm.py has components written in scratch for
CrossEntropyLoss, Adam optimiser, gradient clipping, lr scheduler, get batch, BPE tokenizer.

training_loop.py -> encodes the training data , runs training, saves the model in chekpoints directory, runs inference.

Used A100 GPUs for approx 1hr to train different variations of hyperparameters.
Rented GPUs at ->https://jarvislabs.ai/
model: checkpoints/latest_checkpoint.pth
github link: https://lnkd.in/gKpwTk8H
Below image has some sentences at bottom which was generated after doing 250 epochs/steps.
