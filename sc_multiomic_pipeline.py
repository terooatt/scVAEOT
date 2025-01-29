import torch
import numpy as np
import umap
import ot
from sklearn.preprocessing import normalize
import scanpy as sc
import anndata
import matplotlib.pyplot as plt

# Step 1: Preprocess data (scATAC and scRNA)
def preprocess_data(scatac_data, scrna_data):
    # Normalize scATAC-seq data (TF-IDF normalization)
    scatac_data = normalize(scatac_data, norm='l1', axis=1)
    scatac_data = scatac_data * 1e4
    scatac_data = np.log1p(scatac_data)

    # Normalize scRNA-seq data (log normalization)
    scrna_data = normalize(scrna_data, norm='l1', axis=1)
    scrna_data = scrna_data * 1e4
    scrna_data = np.log1p(scrna_data)

    return scatac_data, scrna_data

# Step 2: Define VAE model
class VAE(torch.nn.Module):
    def __init__(self, input_dim1, input_dim2, latent_dim):
        super(VAE, self).__init__()
        self.encoder1 = torch.nn.Sequential(
            torch.nn.Linear(input_dim1, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, latent_dim)
        )
        self.encoder2 = torch.nn.Sequential(
            torch.nn.Linear(input_dim2, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, latent_dim)
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, input_dim1)
        )
        self.latent_dim = latent_dim

    def forward(self, x1, x2):
        z1 = self.encoder1(x1)
        z2 = self.encoder2(x2)
        return z1, z2

# Step 3: Train VAE
def train_vae(scatac_data, scrna_data, latent_dim=50, epochs=100, lr=0.001):
    vae = VAE(input_dim1=scatac_data.shape[1], input_dim2=scrna_data.shape[1], latent_dim=latent_dim)
    optimizer = torch.optim.Adam(vae.parameters(), lr=lr)

    for epoch in range(epochs):
        z1, z2 = vae(torch.tensor(scatac_data, dtype=torch.float32),
                     torch.tensor(scrna_data, dtype=torch.float32))
        loss = torch.mean((z1 - z2) ** 2)  # Simple MSE loss for alignment
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return vae

# Step 4: Optimal Transport alignment
def align_data(z1, z2):
    cost_matrix = ot.dist(z1.detach().numpy(), z2.detach().numpy())
    ot_plan = ot.emd(ot.unif(z1.shape[0]), ot.unif(z2.shape[0]), cost_matrix)
    aligned_z1 = ot_plan.T @ z2.detach().numpy()
    return aligned_z1

# Step 5: UMAP visualization
def visualize_umap(aligned_z1, z2, labels):
    combined_latent = np.vstack([aligned_z1, z2.detach().numpy()])
    umap_embedding = umap.UMAP().fit_transform(combined_latent)

    plt.scatter(umap_embedding[:, 0], umap_embedding[:, 1], c=labels, s=1)
    plt.savefig("umap_plot.png")
    plt.close()

# Main function
def main():
    # Load data (replace with your data loading logic)
    scatac_data = np.random.rand(100, 2000)  # Example scATAC data
    scrna_data = np.random.rand(100, 3000)   # Example scRNA data
    labels = np.random.randint(0, 5, 200)    # Example labels for UMAP

    # Preprocess data
    scatac_data, scrna_data = preprocess_data(scatac_data, scrna_data)

    # Train VAE
    vae = train_vae(scatac_data, scrna_data)

    # Get latent representations
    z1, z2 = vae(torch.tensor(scatac_data, dtype=torch.float32),
                 torch.tensor(scrna_data, dtype=torch.float32))

    # Align data with optimal transport
    aligned_z1 = align_data(z1, z2)

    # Visualize with UMAP
    visualize_umap(aligned_z1, z2, labels)

if __name__ == "__main__":
    main()
