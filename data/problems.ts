import { Problem } from '../types';

export const PROBLEM_LIBRARY: Problem[] = [
  // Supervised Learning
  {
    id: 'sl-1',
    title: 'Logistic Regression Gradient Descent',
    category: 'Supervised Learning',
    difficulty: 'Medium',
    description: 'Implement the gradient update step for Logistic Regression using binary cross-entropy loss. Assume X is (N, D) and y is (N, 1).',
    examples: [
      { input: 'X=[[1,2],[3,4]], y=[[0],[1]], weights=[0.1, 0.1], lr=0.01', output: 'Updated weights vector' }
    ],
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
    """
    X: (N, D) numpy array
    y: (N, 1) numpy array
    weights: (D, 1) numpy array
    lr: learning rate scalar
    """
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
      { input: 'points=[[1,1],[2,2],[10,10]], query=[1.2, 1.2], k=1', output: '[[1,1]]' }
    ],
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
    # Calculate Euclidean distances
    # Using broadcasting: (N, D) - (D,) -> (N, D)
    diff = points - query
    distances = np.sqrt(np.sum(diff**2, axis=1))
    
    # Get indices of k smallest distances
    # argsort returns indices that would sort the array
    nearest_indices = np.argsort(distances)[:k]
    
    return points[nearest_indices]`
  },

  // Unsupervised Learning
  {
    id: 'ul-1',
    title: 'K-Means Clustering Step',
    category: 'Unsupervised Learning',
    difficulty: 'Medium',
    description: 'Implement the assignment and update steps of K-Means. Given centroids and points, assign points to clusters and recompute centroids.',
    examples: [
      { input: 'points NxD, centroids KxD', output: 'New Centroids KxD' }
    ],
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
      { input: 'X (NxD), k=2', output: 'Components (2xD)' }
    ],
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

  // Deep Learning
  {
    id: 'dl-1',
    title: 'Self-Attention Mechanism',
    category: 'Deep Learning',
    difficulty: 'Hard',
    description: 'Implement the scaled dot-product attention mechanism: softmax(QK^T / sqrt(d_k))V.',
    examples: [
      { input: 'Q, K, V matrices of shape (B, Seq, Dim)', output: 'Attention Output (B, Seq, Dim)' }
    ],
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
        self.input = x
        return np.maximum(0, x)
    
    def backward(self, grad_output):
        """
        grad_output: Gradient flowing from the next layer
        """
        grad_input = grad_output.copy()
        
        # Gradient is 0 where input <= 0
        grad_input[self.input <= 0] = 0
        
        return grad_input`
  },

  // NLP
  {
    id: 'nlp-1',
    title: 'Positional Encoding',
    category: 'NLP',
    difficulty: 'Medium',
    description: 'Implement the sinusoidal positional encoding used in Transformers.',
    examples: [
      { input: 'seq_len=10, d_model=512', output: 'Matrix (10, 512)' }
    ],
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

  // Computer Vision
  {
    id: 'cv-1',
    title: 'Intersection over Union (IoU)',
    category: 'Computer Vision',
    difficulty: 'Medium',
    description: 'Calculate the IoU score for two bounding boxes defined as [x1, y1, x2, y2].',
    examples: [
      { input: 'box1=[0,0,2,2], box2=[1,1,3,3]', output: '0.1428 (Area of intersection / Area of union)' }
    ],
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
      { input: 'Image 4x4, Kernel 3x3', output: 'Output 2x2' }
    ],
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

  // Reinforcement Learning
  {
    id: 'rl-1',
    title: 'Q-Learning Update',
    category: 'Reinforcement Learning',
    difficulty: 'Easy',
    description: 'Write the function to update a Q-table entry given state s, action a, reward r, next state s_next, learning rate alpha, and discount gamma.',
    examples: [
      { input: 'current_q, reward=1, max_next_q=0.5', output: 'updated_q' }
    ],
    hints: ['Q(s,a) = Q(s,a) + alpha * (r + gamma * max(Q(s\', a\')) - Q(s,a))'],
    starterCode: `def update_q_table(q_table, state, action, reward, next_state, alpha, gamma):
    """
    q_table: dict or 2D array mapping (state, action) -> value
    """
    # Your code here
    pass`,
    solution: `def update_q_table(q_table, state, action, reward, next_state, alpha, gamma):
    """
    q_table: dict or 2D array mapping (state, action) -> value
    """
    # Get current Q value
    current_q = q_table[state][action]
    
    # Get max Q value for next state
    # If next_state is terminal, max_next_q is 0
    max_next_q = np.max(q_table[next_state])
    
    # Compute target
    target = reward + gamma * max_next_q
    
    # Update rule
    new_q = current_q + alpha * (target - current_q)
    
    q_table[state][action] = new_q
    
    return q_table`
  },

  // Reasoning
  {
    id: 'rs-1',
    title: 'MCTS Selection (UCT)',
    category: 'Reasoning',
    difficulty: 'Hard',
    description: 'Implement the UCB1 formula to select the best child node in the selection phase of Monte Carlo Tree Search.',
    examples: [
      { input: 'Parent visits, Child visits list, Child wins list', output: 'Index of selected child' }
    ],
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
  }
];