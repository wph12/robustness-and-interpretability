import torch
import logging
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np

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
    logger.info(f"Mean Alignment (Saliency): {np.mean(cosine_similarities_sal):.4f}")
    logger.info(f"Median Alignment (saliency): {np.median(cosine_similarities_sal):.4f}")


    #log ig
    logger.info(f"Mean Alignment (IG): {np.mean(cosine_similarities_ig):.4f}")
    logger.info(f"Median Alignment(IG): {np.median(cosine_similarities_ig):.4f}")


def sal_metrics(model, dataloader, run_id, device, use_infidelity=False, use_max_sensitivity=False, use_sparseness= False):
    model.zero_grad()
    sal = Saliency(model)
    logger = logging.getLogger(run_id)

    #init metrics: infidelity(faithfulness), max-sensitivity(robustness), sparsity(complexity)
    if(use_max_sensitivity):
        max_sensitivity = quantus.MaxSensitivity(nr_samples=100,
        lower_bound=0.1,
        norm_numerator=quantus.norm_func.fro_norm,
        norm_denominator=quantus.norm_func.fro_norm,
        perturb_func=quantus.perturb_func.uniform_noise,
        similarity_func=quantus.similarity_func.difference,
        normalise= False)

    if(use_infidelity):
        infidelity = quantus.Infidelity(
            perturb_baseline="uniform",
            perturb_func=quantus.perturb_func.baseline_replacement_by_indices,
            n_perturb_samples=10,
        )

    if(use_sparseness):
        sparseness = quantus.Sparseness()


    sal_results = {
        'infidelity': [],
        'max_sensitivity': [],
        'sparseness': []
    }


    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        # Compute attributions
        sal_attributions = sal.attribute(
            images, target = labels
        ).sum(axis=1).cpu().numpy()

        images, labels = images.cpu().numpy(), labels.cpu().numpy()

        if(use_infidelity):
            sal_results['infidelity'].extend(
                infidelity(model=model,
                        x_batch=images,
                        y_batch=labels,
                        a_batch=sal_attributions)
            )
        if(use_max_sensitivity):
            sal_results['max_sensitivity'].extend(
                max_sensitivity(model=model,
                                x_batch=images,
                                y_batch=labels,
                                a_batch=sal_attributions,
                                explain_func=quantus.explain,
                                explain_func_kwargs={"method": "Saliency", "softmax": False})
            )
        if(use_sparseness):
            sal_results['sparseness'].extend(
                sparseness(model=model,
                        x_batch=images,
                        y_batch=labels,
                        a_batch=sal_attributions)
            )


    logger.info("NO. OF RESULTS:" + str(len(sal_results['infidelity'])))

    logger.info("######## SALIENCY ##########")
    if(use_infidelity):
        logger.info(f"Mean infidelity (Saliency): {np.mean(sal_results['infidelity']):.4f}")
        logger.info(f"Median infidelity (Saliency): {np.median(sal_results['infidelity']):.4f}")

    if(use_max_sensitivity):
        logger.info(f"Mean max-sensitivity (Saliency): {np.mean(sal_results['max_sensitivity']):.4f}")
        logger.info(f"Median max-sensitivity (Saliency): {np.median(sal_results['max_sensitivity']):.4f}")

    if(use_sparseness):
        logger.info(f"Mean sparseness (Saliency): {np.mean(sal_results['sparseness']):.4f}")
        logger.info(f"Median sparseness (Saliency): {np.median(sal_results['sparseness']):.4f}")



def ig_metrics(model, dataloader, device, run_id, use_infidelity=False, use_max_sensitivity=False, use_sparseness= False):
    model.zero_grad()
    ig = IntegratedGradients(model)
    logger = logging.getLogger(run_id)

    #init metrics: infidelity(faithfulness), max-sensitivity(robustness), sparsity(complexity)
    if(use_max_sensitivity):
        max_sensitivity = quantus.MaxSensitivity(nr_samples=100,
        lower_bound=0.1,
        norm_numerator=quantus.norm_func.fro_norm,
        norm_denominator=quantus.norm_func.fro_norm,
        perturb_func=quantus.perturb_func.uniform_noise,
        similarity_func=quantus.similarity_func.difference,
        normalise= False)

    if(use_infidelity):
        infidelity = quantus.Infidelity(
            perturb_baseline="uniform",
            perturb_func=quantus.perturb_func.baseline_replacement_by_indices,
            n_perturb_samples=10,
        )

    if(use_sparseness):
        sparseness = quantus.Sparseness()


    ig_results = {
        'infidelity': [],
        'max_sensitivity': [],
        'sparseness': []
    }

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        # Compute attributions
        ig_attributions = ig.attribute(
            images, baselines=torch.zeros_like(images), target=labels
        ).cpu().numpy()

        images, labels = images.cpu().numpy(), labels.cpu().numpy()

        if(use_infidelity):
            ig_results['infidelity'].extend(
            infidelity(model=model,
                        x_batch=images,
                        y_batch=labels,
                        a_batch=ig_attributions,
                        explain_func=quantus.explain,
                        explain_func_kwargs={"method": "IntegratedGradients", "softmax": False})
        )
        if(use_max_sensitivity):
            ig_results['max_sensitivity'].extend(
            max_sensitivity(model=model,
                            x_batch=images,
                            y_batch=labels,
                            a_batch=ig_attributions)
        )
        if(use_sparseness):
            ig_results['sparseness'].extend(
            sparseness(model=model,
                       x_batch=images,
                       y_batch=labels,
                       a_batch=ig_attributions)
        )


    logger.info("NO. OF RESULTS:" + str(len(ig_results['infidelity'])))

    logger.info("######## INTEGRATED GRADIENTS ##########")
    logger.info(f"Mean infidelity (Integrated Gradients): {np.mean(ig_results['infidelity']):.4f}")
    logger.info(f"Median infidelity (Integrated Gradients): {np.median(ig_results['infidelity']):.4f}")
    logger.info(f"Mean max-sensitivity (Integrated Gradients): {np.mean(ig_results['max_sensitivity']):.4f}")
    logger.info(f"Median max-sensitivity (Integrated Gradients): {np.median(ig_results['max_sensitivity']):.4f}")
    logger.info(f"Mean sparseness (Integrated Gradients): {np.mean(ig_results['sparseness']):.4f}")
    logger.info(f"Median sparseness (Integrated Gradients): {np.median(ig_results['sparseness']):.4f}")
