Midterm Project
Skip-gram Word2Vec From Scratch

1. Project overview
This project implements a complete Skip-gram Word2Vec model from scratch using NumPy.
The model is trained on the fixed 8-sentence corpus from the assignment and includes:
- corpus preprocessing
- vocabulary construction
- center-context pair generation
- numerically stable softmax
- cross-entropy loss
- forward pass
- backward pass
- SGD training with learning-rate decay
- numerical gradient checking
- cosine-similarity evaluation
- nearest-neighbor retrieval
- comparison against Gensim Word2Vec

The implementation follows the assignment settings:
- embedding dimension d = 10
- context window W = 2
- initial learning rate = 0.025
- learning-rate decay = 0.005
- epochs = 100
- random seed = 0 for the baseline model

2. Repository files
skipgram.py
- Main implementation of the Skip-gram model.
- Contains preprocessing, model definition, gradient checking, training loop, loss plotting, and task-by-task console output for Parts 1 to 3.

evaluate.py
- Evaluation script for Part 4.
- Computes cosine similarities, nearest neighbors, Gensim comparison, and Spearman rank correlation.
- Prints an extended Task 4.3 comparison table with custom scores, Gensim scores, rank positions, Spearman rho, p-value, and two technical differences.

loss_curve.png
- Training loss figure produced by skipgram.py.

skipgram_output.txt
- Saved terminal output from running skipgram.py.

evaluate_output.txt
- Saved terminal output from running evaluate.py.


3. Environment requirements
Recommended Python version:
- Python 3.12

Required packages:
- numpy
- matplotlib
- gensim
- scipy
- argparse

4. Setup
Create and activate a virtual environment:
python3.12 -m venv .venv
source .venv/bin/activate

Install dependencies:
python -m pip install numpy matplotlib gensim scipy

5. How to run
Run the main implementation:
python skipgram.py --output-dir .

This script will:
- tokenize the corpus
- build the sorted vocabulary
- generate training pairs
- verify softmax/loss and backward-pass values
- run numerical gradient checks
- train the baseline model for 100 epochs
- run the two extra hyperparameter experiments
- generate loss_curve.png
- print outputs grouped by task

Run the evaluation script:
python evaluate.py --output-dir .

This script will:
- load the trained baseline pipeline
- compute cosine similarities for the required word pairs
- find top-3 nearest neighbors for the required query words
- train a Gensim Skip-gram model on the same corpus
- compare custom similarities with Gensim similarities
- compute Spearman rho and p-value
- print an extended similarity table for Task 4.3 with custom rank and Gensim rank columns
- print outputs grouped by task

Part 4 summary:
- Task 4.1 evaluates semantic similarity using cosine similarity on the learned W_in embeddings.
- Task 4.2 retrieves top-3 nearest neighbors for the required query words using cosine similarity.
- Task 4.3 compares the custom model against Gensim, reports Spearman rho and p-value, and explains differences due to full softmax vs negative sampling and different training internals.

6. Expected generated outputs
After running skipgram.py:
- loss_curve.png
- console output for Task 1.4, Task 2.1, Task 2.2, Task 2.3, Task 2.4, Task 3.1, and Task 3.3

After running evaluate.py:
- console output for Task 4.1, Task 4.2, and Task 4.3

If you want to save the outputs to text files:
python skipgram.py --output-dir . | tee skipgram_output.txt
python evaluate.py --output-dir . | tee evaluate_output.txt
