import torch
import logging
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F

from torchvision import models

from captum.attr import IntegratedGradients, Saliency #maybe add kernelSHAP?
import quantus


def alignment(model, dataloader, run_id, device):
    logger = logging.getLogger(run_id)

    cosine_similarities_sal = []
    cosine_similarities_ig = []
    model.zero_grad()
    sal = Saliency(model)
    ig = IntegratedGradients(model)

    for images, labels in dataloader:

        images = images.to(device)
        labels = labels.to(device)

        sal_attributions = sal.attribute(
            images, target = labels
        )
        
        ig_attributions = ig.attribute(
            images, baselines=torch.zeros_like(images), target=labels
        )

        
        #Compute Cosine Similarity (saliency)
        similarity_sal = torch.abs(F.cosine_similarity(images, sal_attributions, dim=1)) 
        cosine_similarities_sal.extend(similarity_sal.tolist())

        #Compute Cosine Similarity (IG)
        similarity_ig = torch.abs(F.cosine_similarity(images, ig_attributions, dim=1)) 
        cosine_similarities_ig.extend(similarity_ig.tolist())

    #log saliency
    mean_similarity_sal = cosine_similarities_sal.mean().item()
    logger.info(f"Mean Alignment (Saliency): {mean_similarity_sal:.4f}")
    median_similarity_sal = cosine_similarities_sal.median().item()
    logger.info(f"Median Alignment (saliency): {median_similarity_sal:.4f}")


    #log ig
    mean_similarity_ig = cosine_similarities_ig.mean().item()
    logger.info(f"Mean Alignment (IG): {mean_similarity_ig:.4f}")
    median_similarity_ig = cosine_similarities_ig.median().item()
    logger.info(f"Median Alignment(IG): {median_similarity_ig:.4f}")


def other_metrics(model, dataloader, run_id, device):
    logger = logging.getLogger(run_id)
    model.zero_grad()
    sal = Saliency(model)
    ig = IntegratedGradients(model)

    #init metrics: infidelity(faithfulness), max-sensitivity(robustness), sparsity(complexity)


    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        # Compute attributions
        sal_attributions = sal.attribute(
            images, target = labels
        )
        
        ig_attributions = ig.attribute(
            images, baselines=torch.zeros_like(images), target=labels
        )

        xai_methods = {
            "Saliency": sal_attributions,
            "IntegratedGradients": ig_attributions
        }

        results = quantus.evaluate(
        metrics = metrics,
        xai_methods=xai_methods,
        model=model,
        x_batch=images,
        y_batch=labels,
        **{"softmax": False,}
    )
    
