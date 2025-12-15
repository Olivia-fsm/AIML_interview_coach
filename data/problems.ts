import { Problem } from '../types';

export const PROBLEM_LIBRARY: Problem[] = [
  // --- Supervised Learning ---
  {
    id: 'sl-1',
    title: 'Logistic Regression Gradient Descent',
    category: 'Supervised Learning',
    difficulty: 'Medium',
    description: 'Implement the gradient update step for Logistic Regression using binary cross-entropy loss. Assume X is (N, D) and y is (N, 1).',
    examples: [
      { input: 'X=[[1,2],[3,4]], y=[[0],[1]], weights=[0.1, 0.1], lr=0.01', output: 'weights=[0.102, 0.101] (Approx)' }
    ],
    hiddenTestCase: { input: 'X=[[0,0],[0,0]], y=[[1],[0]], weights=[0.0, 0.0], lr=0.1', output: 'Updated weights handling zeros correctly' },
    hints: ['Remember the sigmoid function.', 'Gradient is X.T * (y_pred - y)'],
    starterCode: `import numpy as np

def update_weights(X, y, weights, lr):
    """
    X: (N, D) numpy array
    y: (N, 1) numpy array
    weights: (D, 1) numpy array
    lr: learning rate scalar
    """
    # Your code here
    pass`,
    solution: `import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def update_weights(X, y, weights, lr):
    X = np.array(X)
    y = np.array(y)
    weights = np.array(weights)
    
    N = X.shape[0]
    
    # Forward pass
    z = np.dot(X, weights)
    y_pred = sigmoid(z)
    
    # Gradient calculation
    # dL/dw = (1/N) * X.T @ (y_pred - y)
    gradient = (1/N) * np.dot(X.T, (y_pred - y))
    
    # Update rule
    new_weights = weights - lr * gradient
    
    return new_weights`
  },
  {
    id: 'sl-2',
    title: 'K-Nearest Neighbors (KNN)',
    category: 'Supervised Learning',
    difficulty: 'Easy',
    description: 'Implement a function to find the K nearest neighbors of a query point from a dataset using Euclidean distance. Do not use sklearn.',
    examples: [
      { input: 'points=[[1,1],[2,2],[10,10]], query=[1.2, 1.2], k=1', output: '[[1, 1]]' }
    ],
    hiddenTestCase: { input: 'points=[[0,0],[0,0],[1,1]], query=[0,0], k=2', output: '[[0,0],[0,0]]' },
    hints: ['Compute distances to all points.', 'Sort arguments by distance.'],
    starterCode: `import numpy as np

def get_k_nearest_neighbors(points, query, k):
    """
    points: (N, D) numpy array
    query: (D,) numpy array
    k: int
    """
    # Your code here
    pass`,
    solution: `import numpy as np

def get_k_nearest_neighbors(points, query, k):
    """
    points: (N, D) numpy array
    query: (D,) numpy array
    k: int
    """
    points = np.array(points)
    query = np.array(query)
    
    # Calculate Euclidean distances
    # Using broadcasting: (N, D) - (D,) -> (N, D)
    diff = points - query
    distances = np.sqrt(np.sum(diff**2, axis=1))
    
    # Get indices of k smallest distances
    # argsort returns indices that would sort the array
    nearest_indices = np.argsort(distances)[:k]
    
    return points[nearest_indices]`
  },

  // --- Unsupervised Learning ---
  {
    id: 'ul-1',
    title: 'K-Means Clustering Step',
    category: 'Unsupervised Learning',
    difficulty: 'Medium',
    description: 'Implement the assignment and update steps of K-Means. Given centroids and points, assign points to clusters and recompute centroids.',
    examples: [
      { input: 'points=[[0,0], [2,2], [0,1], [2,1]], centroids=[[0,0], [2,2]]', output: 'new_centroids=[[0, 0.5], [2, 1.5]], labels=[0, 1, 0, 1]' }
    ],
    hiddenTestCase: { input: 'points=[[1,1],[1,1]], centroids=[[0,0],[10,10]]', output: 'new_centroids=[[1, 1], [10, 10]], labels=[0, 0]' },
    hints: ['Broadcasting can help compute pairwise distances efficiently.', 'Use mean along axis 0 for updates.'],
    starterCode: `import numpy as np

def kmeans_step(points, centroids):
    """
    points: (N, D)
    centroids: (K, D)
    """
    # Your code here
    pass`,
    solution: `import numpy as np

def kmeans_step(points, centroids):
    """
    points: (N, D)
    centroids: (K, D)
    """
    points = np.array(points)
    centroids = np.array(centroids)
    
    # 1. Assignment Step
    # Calculate distances from every point to every centroid
    # shape: (N, 1, D) - (1, K, D) -> (N, K, D)
    distances = np.sqrt(np.sum((points[:, np.newaxis] - centroids[np.newaxis, :])**2, axis=2))
    
    # Assign to nearest centroid (N,)
    labels = np.argmin(distances, axis=1)
    
    # 2. Update Step
    K = centroids.shape[0]
    new_centroids = np.zeros_like(centroids)
    
    for k in range(K):
        # Get points belonging to cluster k
        cluster_points = points[labels == k]
        
        if len(cluster_points) > 0:
            new_centroids[k] = np.mean(cluster_points, axis=0)
        else:
            # Handle empty cluster (e.g., keep old centroid)
            new_centroids[k] = centroids[k]
            
    return new_centroids, labels`
  },
  {
    id: 'ul-2',
    title: 'PCA from Covariance',
    category: 'Unsupervised Learning',
    difficulty: 'Hard',
    description: 'Implement Principal Component Analysis given a data matrix X. Return the top k principal components.',
    examples: [
      { input: 'X=[[1,2], [3,4], [5,6]], k=1', output: 'components=[[-0.42, -0.90]] (Top 1 Eigenvector)' }
    ],
    hiddenTestCase: { input: 'X=Identity(4), k=1', output: 'Any unit vector is valid, check implementation robustness' },
    hints: ['Center the data first.', 'Compute Covariance Matrix.', 'Use np.linalg.eigh or svd.'],
    starterCode: `import numpy as np

def pca(X, k):
    """
    X: (N, D) data matrix
    k: number of components
    """
    # Your code here
    pass`,
    solution: `import numpy as np

def pca(X, k):
    """
    X: (N, D) data matrix
    k: number of components
    """
    X = np.array(X)

    # 1. Center the data
    mean = np.mean(X, axis=0)
    X_centered = X - mean
    
    # 2. Compute Covariance Matrix
    # Cov = (1 / (N-1)) * X_centered.T @ X_centered
    # Note: np.cov expects rows as variables, so we transpose X or use rowvar=False
    cov_matrix = np.cov(X_centered, rowvar=False)
    
    # 3. Eigendecomposition
    # eigh is numerically more stable for symmetric matrices (covariance matrices are symmetric)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # 4. Sort eigenvectors by eigenvalues in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    top_k_indices = sorted_indices[:k]
    
    # Get top k components (eigenvectors are columns in numpy's output)
    components = eigenvectors[:, top_k_indices]
    
    return components`
  },

  // --- Deep Learning ---
  {
    id: 'dl-1',
    title: 'Self-Attention Mechanism',
    category: 'Deep Learning',
    difficulty: 'Hard',
    description: 'Implement the scaled dot-product attention mechanism: softmax(QK^T / sqrt(d_k))V.',
    examples: [
      { input: 'Q=np.eye(2)[None,:], K=np.eye(2)[None,:], V=np.ones((1,2,2))', output: '[[[1., 1.], [1., 1.]]]' }
    ],
    hiddenTestCase: { input: 'Q=Zeros, K=Zeros, V=Ones', output: 'Should return Average of V due to uniform softmax' },
    hints: ['Watch out for matrix multiplication dimensions.', 'Apply mask if necessary (optional for basic version).'],
    starterCode: `import numpy as np

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q, K, V: (Batch, Seq_Len, D_k)
    """
    # Your code here
    pass`,
    solution: `import numpy as np

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q, K, V: (Batch, Seq_Len, D_k)
    """
    Q = np.array(Q)
    K = np.array(K)
    V = np.array(V)
    if mask is not None: mask = np.array(mask)

    d_k = Q.shape[-1]
    
    # 1. Matmul Q and K^T
    # (B, S, D) @ (B, D, S) -> (B, S, S)
    scores = np.matmul(Q, K.swapaxes(-1, -2))
    
    # 2. Scale
    scores = scores / np.sqrt(d_k)
    
    # 3. Mask (Optional)
    if mask is not None:
        scores += (mask * -1e9)
        
    # 4. Softmax
    attention_weights = softmax(scores)
    
    # 5. Matmul with V
    # (B, S, S) @ (B, S, D) -> (B, S, D)
    output = np.matmul(attention_weights, V)
    
    return output`
  },
  {
    id: 'dl-2',
    title: 'ReLU and its Gradient',
    category: 'Deep Learning',
    difficulty: 'Easy',
    description: 'Implement the forward and backward pass for the ReLU activation function.',
    examples: [
      { input: 'x = [-1, 0, 2]', output: 'Forward: [0, 0, 2], Backward gradient mask' }
    ],
    hiddenTestCase: { input: 'x = [-1000, 1000]', output: 'Forward: [0, 1000]' },
    hints: ['ReLU(x) = max(0, x)', 'Gradient is 1 where x > 0, else 0'],
    starterCode: `import numpy as np

class ReLU:
    def forward(self, x):
        # Your code here
        pass
    
    def backward(self, grad_output):
        # Your code here
        pass`,
    solution: `import numpy as np

class ReLU:
    def forward(self, x):
        self.input = np.array(x)
        return np.maximum(0, self.input)
    
    def backward(self, grad_output):
        """
        grad_output: Gradient flowing from the next layer
        """
        grad_input = np.array(grad_output).copy()
        
        # Gradient is 0 where input <= 0
        grad_input[self.input <= 0] = 0
        
        return grad_input`
  },

  // --- NLP (including Search/Ranking) ---
  {
    id: 'nlp-1',
    title: 'Positional Encoding',
    category: 'NLP',
    difficulty: 'Medium',
    description: 'Implement the sinusoidal positional encoding used in Transformers.',
    examples: [
      { input: 'seq_len=2, d_model=4', output: '[[0.,0.,0.,0.], [0.84,0.54,0.,1.]] (Approx)' }
    ],
    hiddenTestCase: { input: 'seq_len=1, d_model=4', output: 'Single vector with correct sin/cos values' },
    hints: ['Use sine for even indices, cosine for odd indices.', 'Div term is 10000^(2i/d_model).'],
    starterCode: `import numpy as np

def positional_encoding(seq_len, d_model):
    """
    Returns (seq_len, d_model) matrix
    """
    # Your code here
    pass`,
    solution: `import numpy as np

def positional_encoding(seq_len, d_model):
    """
    Returns (seq_len, d_model) matrix
    """
    # Initialize matrix
    pe = np.zeros((seq_len, d_model))
    
    # Create positions (0 to seq_len-1) column vector
    position = np.arange(seq_len)[:, np.newaxis]
    
    # Create division term for frequencies
    # 10000^(2i/d_model)
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    
    # Apply Sine to even indices
    pe[:, 0::2] = np.sin(position * div_term)
    
    # Apply Cosine to odd indices
    pe[:, 1::2] = np.cos(position * div_term)
    
    return pe`
  },
  {
    id: 'nlp-2',
    title: 'TF-IDF Calculation',
    category: 'NLP',
    difficulty: 'Medium',
    description: 'Calculate the TF-IDF matrix for a list of sentences manually (without sklearn).',
    examples: [
      { input: '["apple banana", "apple"]', output: 'Sparse matrix or dense array of scores' }
    ],
    hiddenTestCase: { input: '["a", "b", "c"]', output: 'Should have equal weights/dimensions' },
    hints: ['TF = count/total_words_in_doc', 'IDF = log(total_docs / docs_with_term)'],
    starterCode: `import numpy as np
from collections import Counter

def compute_tfidf(corpus):
    """
    corpus: list of string sentences
    """
    # Your code here
    pass`,
    solution: `import numpy as np
import math
from collections import Counter

def compute_tfidf(corpus):
    # 1. Build Vocabulary
    docs = [doc.split() for doc in corpus]
    vocab = sorted(list(set(word for doc in docs for word in doc)))
    word_to_idx = {w: i for i, w in enumerate(vocab)}
    
    N = len(docs)
    V = len(vocab)
    
    # 2. Compute IDF
    # Number of docs containing word w
    doc_freq = Counter()
    for doc in docs:
        doc_words = set(doc)
        for word in doc_words:
            doc_freq[word] += 1
            
    idf = np.zeros(V)
    for word, idx in word_to_idx.items():
        # Add 1 to avoid division by zero (smoothing)
        idf[idx] = math.log((N + 1) / (doc_freq[word] + 1)) + 1
        
    # 3. Compute TF and TF-IDF
    tfidf_matrix = np.zeros((N, V))
    
    for i, doc in enumerate(docs):
        word_counts = Counter(doc)
        total_words = len(doc)
        
        for word, count in word_counts.items():
            tf = count / total_words
            idx = word_to_idx[word]
            tfidf_matrix[i, idx] = tf * idf[idx]
            
    return tfidf_matrix, vocab`
  },
  {
    id: 'nlp-3',
    title: 'NDCG Score (Ranking)',
    category: 'NLP',
    difficulty: 'Medium',
    description: 'Implement Normalized Discounted Cumulative Gain (NDCG) at K. This is a primary metric for Ranking and Recommendation systems.',
    examples: [
      { input: 'relevance=[3, 2, 3, 0, 1, 2], k=3', output: '0.94 (Approx)' }
    ],
    hiddenTestCase: { input: 'relevance=[0,0,0], k=3', output: '0.0' },
    hints: ['DCG = sum(rel / log2(i + 1)).', 'IDCG = DCG of ideal (sorted) relevance.', 'NDCG = DCG / IDCG.'],
    starterCode: `import numpy as np

def ndcg_at_k(relevance, k):
    """
    relevance: list of true relevance scores (e.g., [3, 0, 1]) matching the predicted order
    k: top k to consider
    """
    # Your code here
    pass`,
    solution: `import numpy as np

def ndcg_at_k(relevance, k):
    """
    relevance: list of true relevance scores in the predicted order
    k: top k to consider
    """
    relevance = np.asfarray(relevance)[:k]
    if relevance.size == 0:
        return 0.0
        
    # 1. Calculate DCG
    # Formula: sum( (2^rel - 1) / log2(i + 2) ) or simple sum( rel / log2(i + 2) )
    # Using standard log2(i+2) formulation where i starts at 0 (rank 1 is i=0 -> log2(2)=1)
    
    discounts = np.log2(np.arange(len(relevance)) + 2)
    dcg = np.sum(relevance / discounts)
    
    # 2. Calculate IDCG (Ideal DCG)
    # Sort relevance in descending order to get ideal ranking
    ideal_relevance = sorted(relevance, reverse=True)
    idcg = np.sum(ideal_relevance / discounts)
    
    if idcg == 0:
        return 0.0
        
    return dcg / idcg`
  },
  {
    id: 'nlp-4',
    title: 'BM25 Score (Search)',
    category: 'NLP',
    difficulty: 'Medium',
    description: 'Implement the BM25 scoring function for a single document given a query.',
    examples: [
      { input: 'doc_len=100, avg_dl=80, tf=3, qf=1, N=1000, n=50', output: 'Score (float)' }
    ],
    hiddenTestCase: { input: 'tf=0', output: 'Score should be 0' },
    hints: ['BM25 = IDF * ((TF * (k1 + 1)) / (TF + k1 * (1 - b + b * DL / AVG_DL)))', 'IDF = log((N - n + 0.5) / (n + 0.5) + 1)'],
    starterCode: `import math

def bm25_score(tf, doc_len, avg_dl, N, n, k1=1.5, b=0.75):
    """
    tf: term frequency in doc
    doc_len: length of document
    avg_dl: average document length in corpus
    N: total number of documents
    n: number of documents containing the term
    """
    # Your code here
    pass`,
    solution: `import math

def bm25_score(tf, doc_len, avg_dl, N, n, k1=1.5, b=0.75):
    """
    tf: term frequency in doc
    doc_len: length of document
    avg_dl: average document length in corpus
    N: total number of documents
    n: number of documents containing the term
    """
    # 1. Compute Inverse Document Frequency (IDF)
    # Probabilistic IDF formula often used in Okapi BM25
    idf = math.log((N - n + 0.5) / (n + 0.5) + 1)
    
    # 2. Compute Term Frequency Component
    # Numerator
    num = tf * (k1 + 1)
    
    # Denominator with length normalization
    # 1 - b + b * (doc_len / avg_dl) penalizes long documents
    denom = tf + k1 * (1 - b + b * (doc_len / avg_dl))
    
    return idf * (num / denom)`
  },

  // --- Computer Vision ---
  {
    id: 'cv-1',
    title: 'Intersection over Union (IoU)',
    category: 'Computer Vision',
    difficulty: 'Medium',
    description: 'Calculate the IoU score for two bounding boxes defined as [x1, y1, x2, y2].',
    examples: [
      { input: 'box1=[0,0,2,2], box2=[1,1,3,3]', output: '0.1428 (Area of intersection / Area of union)' }
    ],
    hiddenTestCase: { input: 'box1=[0,0,1,1], box2=[5,5,6,6]', output: '0.0 (No overlap)' },
    hints: ['Find intersection coordinates: max(x1s), max(y1s), min(x2s), min(y2s).', 'Width = max(0, x2-x1).'],
    starterCode: `def iou(box1, box2):
    """
    box: [x1, y1, x2, y2]
    """
    # Your code here
    pass`,
    solution: `def iou(box1, box2):
    """
    box: [x1, y1, x2, y2]
    """
    # Determine coordinates of intersection rectangle
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
        
    # Area of intersection
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    # Area of union
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - intersection_area
    
    return intersection_area / union_area`
  },
  {
    id: 'cv-2',
    title: '2D Convolution (Naive)',
    category: 'Computer Vision',
    difficulty: 'Medium',
    description: 'Implement a 2D convolution operation (valid padding) given an image and a kernel.',
    examples: [
      { input: 'image=np.ones((4,4)), kernel=np.ones((3,3))', output: '[[9., 9.], [9., 9.]]' }
    ],
    hiddenTestCase: { input: 'Image 3x3, Kernel 3x3', output: 'Output 1x1 (Dot product of entire image)' },
    hints: ['Nested loops iterating over the image dimensions.'],
    starterCode: `import numpy as np

def convolve2d(image, kernel):
    """
    Naive implementation of 2D convolution with valid padding.
    """
    # Your code here
    pass`,
    solution: `import numpy as np

def convolve2d(image, kernel):
    """
    Naive implementation of 2D convolution with valid padding.
    """
    image = np.array(image)
    kernel = np.array(kernel)
    
    H, W = image.shape
    k_h, k_w = kernel.shape
    
    # Valid padding output dimensions
    out_h = H - k_h + 1
    out_w = W - k_w + 1
    
    output = np.zeros((out_h, out_w))
    
    for i in range(out_h):
        for j in range(out_w):
            # Extract region of interest
            region = image[i:i+k_h, j:j+k_w]
            # Element-wise multiplication and sum
            output[i, j] = np.sum(region * kernel)
            
    return output`
  },

  // --- Reinforcement Learning ---
  {
    id: 'rl-1',
    title: 'Q-Learning Update',
    category: 'Reinforcement Learning',
    difficulty: 'Easy',
    description: 'Write the function to update a Q-table entry given state s, action a, reward r, next state s_next, learning rate alpha, and discount gamma.',
    examples: [
      { input: 'q_table={0:{1:0.5}}, state=0, action=1, reward=1, next_state=2, alpha=0.1, gamma=0.9', output: 'updated_q=0.55 (approx)' }
    ],
    hiddenTestCase: { input: 'alpha=0, q_table=Unchanged', output: 'q_table should remain identical' },
    hints: ['Q(s,a) = Q(s,a) + alpha * (r + gamma * max(Q(s\', a\')) - Q(s,a))'],
    starterCode: `def update_q_table(q_table, state, action, reward, next_state, alpha, gamma):
    """
    q_table: dict or 2D array mapping (state, action) -> value
    """
    # Your code here
    pass`,
    solution: `import numpy as np
def update_q_table(q_table, state, action, reward, next_state, alpha, gamma):
    """
    q_table: dict or 2D array mapping (state, action) -> value
    """
    # Get current Q value
    current_q = q_table[state][action]
    
    # Get max Q value for next state
    # If next_state is terminal, max_next_q is 0
    max_next_q = np.max(q_table[next_state]) if next_state in q_table else 0
    
    # Compute target
    target = reward + gamma * max_next_q
    
    # Update rule
    new_q = current_q + alpha * (target - current_q)
    
    q_table[state][action] = new_q
    
    return q_table`
  },
  {
    id: 'rl-2',
    title: 'PPO Clipped Objective',
    category: 'Reinforcement Learning',
    difficulty: 'Hard',
    description: 'Implement the Probabilistic Policy Optimization (PPO) clipped surrogate objective function.',
    examples: [
      { input: 'old_probs=[0.5], new_probs=[0.6], adv=[1.0], eps=0.2', output: 'loss=-1.2 (Minimized neg objective)' }
    ],
    hiddenTestCase: { input: 'ratio=1.0, adv=1.0, eps=0.2', output: '-1.0' },
    hints: ['Loss = -min(ratio*adv, clip(ratio, 1-eps, 1+eps)*adv).', 'Remember we typically minimize negative objective.'],
    starterCode: `import numpy as np

def ppo_loss(old_log_probs, new_log_probs, advantages, epsilon=0.2):
    """
    Calculates the PPO clipped loss.
    All inputs are numpy arrays of shape (Batch,).
    """
    # Your code here
    pass`,
    solution: `import numpy as np

def ppo_loss(old_log_probs, new_log_probs, advantages, epsilon=0.2):
    """
    Calculates the PPO clipped loss.
    """
    old_log_probs = np.array(old_log_probs)
    new_log_probs = np.array(new_log_probs)
    advantages = np.array(advantages)
    
    # 1. Calculate ratios: pi_new / pi_old
    # Since we have log probs, ratio = exp(log_new - log_old)
    ratios = np.exp(new_log_probs - old_log_probs)
    
    # 2. Unclipped part: ratio * advantage
    surr1 = ratios * advantages
    
    # 3. Clipped part: clip(ratio, 1-eps, 1+eps) * advantage
    ratios_clipped = np.clip(ratios, 1 - epsilon, 1 + epsilon)
    surr2 = ratios_clipped * advantages
    
    # 4. Take minimum (element-wise)
    # We want to maximize the objective, so we take min of the "pessimistic" bounds
    objective = np.minimum(surr1, surr2)
    
    # 5. Return negative mean (for minimization)
    return -np.mean(objective)`
  },
  {
    id: 'rl-3',
    title: 'DPO Loss (LLM Alignment)',
    category: 'Reinforcement Learning',
    difficulty: 'Medium',
    description: 'Implement the Direct Preference Optimization (DPO) loss function used for aligning LLMs without a Reward Model.',
    examples: [
      { input: 'pol_chosen=[-1.0], pol_rejected=[-2.0], ref_chosen=[-1.0], ref_rejected=[-2.0]', output: '0.693 (Log Sigmoid of 0)' }
    ],
    hiddenTestCase: { input: 'Equal logps for all', output: 'Standard softplus loss for 0 difference' },
    hints: ['DPO uses a binary cross entropy objective on the implicit reward.', 'log(sigmoid(beta * (log_ratios_chosen - log_ratios_rejected)))'],
    starterCode: `import numpy as np

def dpo_loss(policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps, beta=0.1):
    """
    Computes DPO Loss. Inputs are (Batch,) arrays.
    """
    # Your code here
    pass`,
    solution: `import numpy as np

def dpo_loss(policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps, beta=0.1):
    """
    Computes DPO Loss.
    L_DPO = -E[ log sigmoid ( beta * (r_chosen - r_rejected) ) ]
    where implicit reward r(x,y) = beta * (log pi(y|x) - log pi_ref(y|x))
    """
    policy_chosen_logps = np.array(policy_chosen_logps)
    policy_rejected_logps = np.array(policy_rejected_logps)
    ref_chosen_logps = np.array(ref_chosen_logps)
    ref_rejected_logps = np.array(ref_rejected_logps)

    # 1. Calculate log probability ratios for chosen and rejected
    # log(pi/ref) = log(pi) - log(ref)
    chosen_logratios = policy_chosen_logps - ref_chosen_logps
    rejected_logratios = policy_rejected_logps - ref_rejected_logps
    
    # 2. Calculate logits for the preference
    logits = chosen_logratios - rejected_logratios
    
    # 3. Apply Beta scale
    scaled_logits = beta * logits
    
    # 4. Compute log sigmoid
    # log(1 / (1 + exp(-x))) = -log(1 + exp(-x))
    # Softplus is log(1 + exp(x))
    # Stable implementation: -np.logaddexp(0, -scaled_logits)
    log_sigmoid = -np.logaddexp(0, -scaled_logits)
    
    # 5. Negative log likelihood (Loss)
    loss = -np.mean(log_sigmoid)
    
    return loss`
  },
  {
    id: 'rl-4',
    title: 'GRPO Advantage (DeepSeek R1)',
    category: 'Reinforcement Learning',
    difficulty: 'Hard',
    description: 'Implement the Group Relative Policy Optimization (GRPO) advantage calculation. Given a group of rewards for the same input, normalize them to compute advantages.',
    examples: [
        { input: 'rewards = [1.0, 2.0, 3.0]', output: 'Advantages = [-1.22, 0.0, 1.22]' }
    ],
    hiddenTestCase: { input: 'rewards = [5.0, 5.0]', output: 'Advantages = [0.0, 0.0] (handling std=0)' },
    hints: ['Advantage = (Reward - Mean(Group)) / Std(Group)', 'Compute statistics over the group dimension.'],
    starterCode: `import numpy as np

def compute_grpo_advantages(rewards):
    """
    rewards: List or array of rewards for G outputs generated from the same prompt.
    Returns: Normalized advantages.
    """
    # Your code here
    pass`,
    solution: `import numpy as np

def compute_grpo_advantages(rewards):
    """
    rewards: List or array of rewards for G outputs generated from the same prompt.
    Returns: Normalized advantages.
    """
    rewards = np.array(rewards)
    
    # 1. Compute Mean and Std of the group
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    
    # Avoid division by zero
    if std_reward < 1e-8:
        std_reward = 1.0
        
    # 2. Normalize
    # A_i = (r_i - mean) / std
    advantages = (rewards - mean_reward) / std_reward
    
    return advantages`
  },

  // --- Reasoning ---
  {
    id: 'rs-1',
    title: 'MCTS Selection (UCT)',
    category: 'Reasoning',
    difficulty: 'Hard',
    description: 'Implement the UCB1 formula to select the best child node in the selection phase of Monte Carlo Tree Search.',
    examples: [
      { input: 'parent_visits=100, child_visits=[10, 50], child_wins=[5, 30], c=1.41', output: '0 (Higher UCB score due to exploration)' }
    ],
    hiddenTestCase: { input: 'Parent visits=10, child has 0 visits', output: 'Index of unvisited child' },
    hints: ['score = (wins/visits) + c * sqrt(ln(parent_visits)/visits)'],
    starterCode: `import math
import numpy as np

def uct_select(parent_visits, child_visits, child_wins, c=1.41):
    """
    Selects the child node maximizing UCB1 score.
    child_visits: list or array of visit counts for each child
    child_wins: list or array of win counts (or total reward) for each child
    c: exploration constant
    """
    # Your code here
    pass`,
    solution: `import math
import numpy as np

def uct_select(parent_visits, child_visits, child_wins, c=1.41):
    """
    Selects the child node maximizing UCB1 score.
    child_visits: list or array of visit counts for each child
    child_wins: list or array of win counts (or total reward) for each child
    c: exploration constant
    """
    # Avoid division by zero by adding small epsilon or checking 0 visits
    scores = []
    
    log_parent = math.log(parent_visits)
    
    for i, visits in enumerate(child_visits):
        if visits == 0:
            # If a child hasn't been visited, prioritize it (infinite score)
            return i
            
        exploitation = child_wins[i] / visits
        exploration = c * math.sqrt(log_parent / visits)
        
        scores.append(exploitation + exploration)
        
    return np.argmax(scores)`
  },
  {
    id: 'rs-2',
    title: 'BLEU Score (Evaluation)',
    category: 'Reasoning',
    difficulty: 'Medium',
    description: 'Implement a simplified BLEU score computation for a single sentence pair considering only unigrams (1-gram) precision.',
    examples: [
      { input: 'cand="the cat is on the mat", ref="the cat is on mat"', output: '0.83 (5/6 matches)' }
    ],
    hiddenTestCase: { input: 'cand="the the the", ref="the cat"', output: '1/3 (clipped count is 1)' },
    hints: ['Count occurrences in candidate.', 'Clip count by max occurrences in reference.', 'Divide by total candidate words.'],
    starterCode: `from collections import Counter

def bleu_1_gram(candidate, reference):
    """
    candidate: list of tokens (words)
    reference: list of tokens (words)
    Returns: Unigram precision
    """
    # Your code here
    pass`,
    solution: `from collections import Counter

def bleu_1_gram(candidate, reference):
    """
    candidate: list of tokens (words)
    reference: list of tokens (words)
    Returns: Unigram precision
    """
    cand_counts = Counter(candidate)
    ref_counts = Counter(reference)
    
    correct_matches = 0
    
    for word, count in cand_counts.items():
        # Clipped count: min(candidate_count, reference_count)
        # Prevents generating "the the the" to game the metric
        correct_matches += min(count, ref_counts[word])
        
    if len(candidate) == 0:
        return 0.0
        
    return correct_matches / len(candidate)`
  },
  {
    id: 'rs-3',
    title: 'Beam Search (Inference)',
    category: 'Reasoning',
    difficulty: 'Hard',
    description: 'Implement a beam search step. Given current sequences with scores and a function to get next token probabilities, return the top k new sequences.',
    examples: [
      { input: 'sequences=[([1], 0.5)], k=2', output: 'Top 2 sequences with updated scores' }
    ],
    hiddenTestCase: { input: 'k=1', output: 'Top 1 sequence' },
    hints: ['Expand every current sequence by all possible vocabulary tokens.', 'Score = old_score + log(prob)', 'Select top k from ALL candidates.'],
    starterCode: `import numpy as np

def beam_search_step(sequences, next_token_probs, k):
    """
    sequences: list of tuples (token_list, score)
    next_token_probs: function taking token_list and returning (vocab_size,) log probs
    k: beam width
    """
    # Your code here
    pass`,
    solution: `import numpy as np

def beam_search_step(sequences, next_token_probs, k):
    """
    sequences: list of tuples (token_list, log_score)
    next_token_probs: function taking token_list and returning (vocab_size,) log probs
    k: beam width
    """
    all_candidates = []
    
    # 1. Expand each current sequence
    for seq, score in sequences:
        # Get log probabilities for the next token
        # (Assuming next_token_probs returns array of log-probs)
        log_probs = next_token_probs(seq)
        
        # Create new candidates
        for token_id, log_prob in enumerate(log_probs):
            new_seq = seq + [token_id]
            new_score = score + log_prob
            all_candidates.append((new_seq, new_score))
            
    # 2. Sort all candidates by score (descending)
    ordered = sorted(all_candidates, key=lambda x: x[1], reverse=True)
    
    # 3. Select top k
    return ordered[:k]`
  },
  
  // --- Architecture (New) ---
  {
    id: 'arch-1',
    title: 'MoE Top-K Gating',
    category: 'Architecture',
    difficulty: 'Medium',
    description: 'Implement the routing logic for a Mixture of Experts layer. Given inputs and gate weights, select the top-k experts and compute routing weights.',
    examples: [
        { input: 'x=[[1,0]], w_gate=[[0.8, 0.1, 0.1], [0.1, 0.8, 0.1]], k=1', output: 'indices=[[0]], weights=[[1.0]]' }
    ],
    hiddenTestCase: { input: 'k=1', output: 'Top-1 expert' },
    hints: ['Compute logits = x @ W_gate', 'Select top-k indices', 'Softmax ONLY the top-k values'],
    starterCode: `import numpy as np

def moe_gating(x, w_gate, k=2):
    """
    x: (Batch, HiddenDim)
    w_gate: (HiddenDim, NumExperts)
    Returns: 
      top_k_indices: (Batch, k)
      top_k_weights: (Batch, k) (normalized)
    """
    # Your code here
    pass`,
    solution: `import numpy as np

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

def moe_gating(x, w_gate, k=2):
    """
    x: (Batch, HiddenDim)
    w_gate: (HiddenDim, NumExperts)
    """
    # 1. Compute routing logits
    # (B, H) @ (H, E) -> (B, E)
    logits = np.dot(x, w_gate)
    
    # 2. Find top-k experts
    # argsort gives ascending order, so take last k and reverse
    top_k_indices = np.argsort(logits, axis=1)[:, -k:][:, ::-1]
    
    # 3. Gather top-k logits
    # Advanced indexing to gather values
    rows = np.arange(x.shape[0])[:, np.newaxis]
    top_k_logits = logits[rows, top_k_indices]
    
    # 4. Normalize (Softmax) just the top-k
    top_k_weights = softmax(top_k_logits)
    
    return top_k_indices, top_k_weights`
  },
  {
    id: 'arch-2',
    title: 'Sliding Window Attention Mask',
    category: 'Architecture',
    difficulty: 'Medium',
    description: 'Generate the attention mask for sliding window attention (e.g. Mistral). Each token can only attend to itself and w previous tokens.',
    examples: [
        { input: 'seq_len=5, window_size=2', output: '5x5 Boolean Mask (Diagonals)' }
    ],
    hiddenTestCase: { input: 'window_size=0', output: 'Identity matrix (attend only to self)' },
    hints: ['mask[i, j] is True if i >= j and i - j <= window_size'],
    starterCode: `import numpy as np

def sliding_window_mask(seq_len, window_size):
    """
    Returns (seq_len, seq_len) mask where 1 means attend, 0 means block.
    """
    # Your code here
    pass`,
    solution: `import numpy as np

def sliding_window_mask(seq_len, window_size):
    """
    Returns (seq_len, seq_len) mask where 1 means attend, 0 means block.
    """
    # Create grid of indices
    # i (rows) is query position, j (cols) is key position
    i, j = np.indices((seq_len, seq_len))
    
    # Condition 1: Causal (i >= j)
    # Condition 2: Within window (i - j <= window_size)
    mask = (i >= j) & ((i - j) <= window_size)
    
    return mask.astype(int)`
  },

  // --- System Design (New) ---
  {
    id: 'sys-1',
    title: 'KV Cache Inference',
    category: 'System Design',
    difficulty: 'Medium',
    description: 'Implement a function to update the Key-Value cache during autoregressive decoding.',
    examples: [
        { input: 'new_k=[[1]], new_v=[[2]], cache=([[0]], [[0]])', output: 'updated_k=[[0, 1]], updated_v=[[0, 2]]' }
    ],
    hiddenTestCase: { input: 'past_kv_cache=None', output: 'Returns new_k, new_v' },
    hints: ['Concatenate new keys/values to the existing cache along the sequence dimension.'],
    starterCode: `import numpy as np

def update_kv_cache(new_k, new_v, cache=None):
    """
    new_k, new_v: (Batch, 1, HeadDim) - current step features
    cache: tuple(past_k, past_v) or None. Each is (Batch, PastLen, HeadDim)
    Returns: updated_k, updated_v
    """
    # Your code here
    pass`,
    solution: `import numpy as np

def update_kv_cache(new_k, new_v, cache=None):
    """
    new_k, new_v: (Batch, 1, HeadDim)
    cache: tuple(past_k, past_v) or None
    """
    if cache is None:
        return new_k, new_v
        
    past_k, past_v = cache
    
    # Concatenate along sequence dimension (axis 1)
    updated_k = np.concatenate([past_k, new_k], axis=1)
    updated_v = np.concatenate([past_v, new_v], axis=1)
    
    return updated_k, updated_v`
  },
  {
    id: 'sys-2',
    title: 'Tensor Parallel Linear Layer',
    category: 'System Design',
    difficulty: 'Hard',
    description: 'Simulate the forward pass of a Tensor Parallel MLP block (ColumnParallel -> RowParallel). Assume input x is replicated.',
    examples: [
        { input: 'x=[[1.0]], W_col_shard=[[0.5]], W_row_shard=[[1.0]]', output: '[[0.5]] (Partial sum)' }
    ],
    hiddenTestCase: { input: 'x=zeros', output: 'Output should be zeros' },
    hints: ['ColumnParallel splits output dim. RowParallel splits input dim.', 'Need an AllReduce (sum) after RowParallel to synchronize.'],
    starterCode: `import numpy as np

def tensor_parallel_mlp(x, W_col_shard, W_row_shard):
    """
    Simulates a TP block on a single rank.
    x: (Batch, Dim) - Input is identical on all ranks
    W_col_shard: (Dim, Hidden/WorldSize) - Part of first layer
    W_row_shard: (Hidden/WorldSize, Dim) - Part of second layer
    
    Note: In real TP, we need communication. Here, assume we return the partial result 
    and a note on what communication is needed.
    """
    # Your code here
    pass`,
    solution: `import numpy as np

def tensor_parallel_mlp(x, W_col_shard, W_row_shard):
    """
    x: (Batch, Dim)
    """
    # 1. Column Parallel Layer
    # Each rank computes a slice of the hidden state
    # X @ W_col_part -> (Batch, Hidden/P)
    hidden_slice = np.dot(x, W_col_shard)
    
    # Activation (GeLU/ReLU) applied locally on the slice
    hidden_slice = np.maximum(0, hidden_slice) # Simple ReLU
    
    # 2. Row Parallel Layer
    # Each rank computes a partial sum of the final output
    # H_slice @ W_row_part -> (Batch, Dim)
    output_partial = np.dot(hidden_slice, W_row_shard)
    
    # Note: In a real system, an All-Reduce (Sum) is required here 
    # to aggregate output_partial from all ranks to get the final Output.
    
    return output_partial`
  }
];