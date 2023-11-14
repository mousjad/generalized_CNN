import torch
import torch.nn as nn
import torch.optim as optim

class KNNImputer(nn.Module):
    def __init__(self, k_neighbors=5, max_iter=100, tol=1e-4):
        super(KNNImputer, self).__init__()
        self.k_neighbors = k_neighbors
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X):
        self.X = X
        self.mask = ~torch.isnan(X)
        self.rows, self.cols = X.size()

    def forward(self):
        for _ in range(self.max_iter):
            last_X = self.X.clone()

            # Iterate through each element in the matrix
            for i in range(self.rows):
                for j in range(self.cols):
                    if not self.mask[i, j]:
                        # If the value is missing, impute it
                        self.X[i, j] = self.knn_impute(i, j)

            # Check for convergence
            if torch.sum(torch.abs(self.X - last_X)) < self.tol:
                break

        return self.X

    def knn_impute(self, i, j):
        # Find the k-nearest neighbors
        distances = torch.sum((self.X[self.mask] - self.X[i, j]) ** 2)
        _, indices = torch.topk(distances, self.k_neighbors, largest=False)

        # Use the mean of the k-nearest neighbors to impute the missing value
        return torch.mean(self.X[self.mask][indices])

# Example usage:
# Assuming you have a 2D tensor 'data' with missing values represented as NaN
# and you want to impute those missing values using KNN with k=3.

# Create a KNNImputer instance
imputer = KNNImputer(k_neighbors=3)

# Fit the imputer to your data
data = torch.tensor([[1.0, 2.0, float('nan')], [4.0, float('nan'), 6.0]])

imputer.fit(data)

# Perform imputation
imputed_data = imputer()

# Display the imputed data
print("Imputed Data:\n", imputed_data)
