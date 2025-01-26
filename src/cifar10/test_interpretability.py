import torch
import logging
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F

from torchvision import models

from captum.attr import IntegratedGradients, Saliency

def integrated_gradient(model, dataloader, run_id, device):
    logger = logging.getLogger(run_id)

    cosine_similarities = []
    model.zero_grad()
    ig = IntegratedGradients(model)

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        # Compute attributions
        attributions, _ = ig.attribute(
            images, baselines=torch.zeros_like(images), target=labels, return_convergence_delta=True
        )
        
        # Step 4: Flatten Inputs and Attributions
        images_flat = images.view(images.size(0), -1)  # Flatten each image into 1D
        attributions_flat = attributions.view(attributions.size(0), -1)  # Flatten attributions
        
        # Step 5: Compute Cosine Similarity
        similarity = F.cosine_similarity(images_flat, attributions_flat, dim=1)
        cosine_similarities.extend(similarity.tolist())

    cosine_similarities = torch.tensor(cosine_similarities)
    mean_similarity = cosine_similarities.mean().item()

    print(f"Mean Cosine Similarity (IG): {mean_similarity:.4f}")
    logger.info(f"Mean Cosine Similarity (IG): {mean_similarity:.4f}")

def saliency(model, dataloader, run_id, device):
    logger = logging.getLogger(run_id)

    cosine_similarities = []
    model.zero_grad()
    sal = Saliency(model)

    for images, labels in dataloader:

        images = images.to(device)
        labels = labels.to(device)
        # Compute attributions
        attributions = sal.attribute(
            images, target=labels
        )
        
        # Step 4: Flatten Inputs and Attributions
        images_flat = images.view(images.size(0), -1)  # Flatten each image into 1D
        attributions_flat = attributions.view(attributions.size(0), -1)  # Flatten attributions
        
        # Step 5: Compute Cosine Similarity
        similarity = F.cosine_similarity(images_flat, attributions_flat, dim=1)
        cosine_similarities.extend(similarity.tolist())

    cosine_similarities = torch.tensor(cosine_similarities)
    mean_similarity = cosine_similarities.mean().item()

    print(f"Mean Cosine Similarity (Saliency): {mean_similarity:.4f}")
    logger.info(f"Mean Cosine Similarity (Saliency): {mean_similarity:.4f}")


    median_similarity = cosine_similarities.median().item()
    print(f"Median Cosine Similarity (Saliency): {median_similarity:.4f}")
    logger.info(f"Mean Cosine Similarity (Saliency): {median_similarity:.4f}")