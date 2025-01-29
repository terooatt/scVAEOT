Pipeline Overview
The goal of this pipeline is to integrate two single-cell multi-omic datasets (scATAC-seq and scRNA-seq) from the same sample type. The pipeline uses a Variational Autoencoder (VAE) to learn a shared latent representation of the two modalities and then aligns the datasets using Optimal Transport (OT). Finally, the aligned data is visualized using UMAP.

Pipeline Steps
Data Preprocessing:

Normalize the scATAC-seq and scRNA-seq data.

Apply log transformation to make the data more suitable for downstream analysis.

Ensure both datasets are in a compatible format for integration.

Variational Autoencoder (VAE):

A VAE is trained with two encoders (one for each modality) and a shared latent space.

The VAE learns to map the high-dimensional data from both modalities into a lower-dimensional latent space that captures shared biological features.

Optimal Transport (OT):

Compute a cost matrix between the latent representations of the scATAC-seq and scRNA-seq cells.

Use the Sinkhorn-Knopp algorithm to find the optimal coupling between the two datasets.

Align the scATAC-seq data to the scRNA-seq data in the shared latent space.

UMAP Visualization:

Combine the aligned latent representations of both datasets.

Use UMAP to reduce the combined latent space to 2D for visualization.

Generate a UMAP plot to show the integrated data, colored by cell type or modality.

GitHub Actions Workflow
The GitHub Actions workflow automates the execution of the pipeline whenever changes are pushed to the main branch or a pull request is opened. Here's how it works:

Trigger:

The workflow is triggered by a push to the main branch or a pull_request targeting the main branch.

Steps:

Checkout Repository: The workflow checks out the code from the repository.

Set Up Python: Installs the specified version of Python (e.g., 3.9).

Install Dependencies: Installs required Python packages (torch, numpy, umap-learn, ot, scanpy, matplotlib).

Run Pipeline: Executes the sc_multiomic_pipeline.py script, which performs the data integration and generates a UMAP plot.

Upload UMAP Plot: Saves the UMAP plot (umap_plot.png) as a build artifact, which can be downloaded from the GitHub Actions interface.


.
├── .github
│   └── workflows
│       └── run_pipeline.yml            # GitHub Actions workflow file
├── sc_multiomic_pipeline.py            # Python script for the pipeline
└── README.md                           # Documentation for the repository

