import torch
import logging
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np

from torchvision import models
from sklearn.metrics import auc
from captum.attr import IntegratedGradients, Saliency #maybe add kernelSHAP?
import quantus


def alignment(model, dataloader, run_id, device, use_ig = False):
    logger = logging.getLogger(run_id)

    cosine_similarities_sal = []
    model.zero_grad()
    sal = Saliency(model)
    if(use_ig):
        cosine_similarities_ig = []
        ig = IntegratedGradients(model)

    for images, _ in dataloader:

        images = images.to(device)
        # labels = labels.to(device)

        with torch.no_grad():  # No need to compute gradients during testing
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

        preds = preds.to(device)

        attributions = sal.attribute(
            images, target = preds
        )
        
        images_flat = images.view(images.size(0), -1)

        #Compute Cosine Similarity (saliency)
        sal_flat = attributions.view(attributions.size(0), -1)
        similarity_sal = torch.abs(F.cosine_similarity(images_flat, sal_flat, dim=1)) 
        cosine_similarities_sal.extend(similarity_sal.tolist())

        if(use_ig):
            ig_attributions = ig.attribute(
                images, baselines=torch.zeros_like(images), target=preds
            )
            # Compute Cosine Similarity (IG)
            ig_flat = ig_attributions.view(ig_attributions.size(0), -1)
            similarity_ig = torch.abs(F.cosine_similarity(images_flat, ig_flat, dim=1)) 
            cosine_similarities_ig.extend(similarity_ig.tolist())

    #log saliency
    logger.info(f"Mean Alignment (Saliency): {np.mean(cosine_similarities_sal):.4f}")
    logger.info(f"Median Alignment (saliency): {np.median(cosine_similarities_sal):.4f}")


    #log ig
    if(use_ig):
        logger.info(f"Mean Alignment (IG): {np.mean(cosine_similarities_ig):.4f}")
        logger.info(f"Median Alignment(IG): {np.median(cosine_similarities_ig):.4f}")


def interpretability_metrics(model, dataloader, run_id, xai_method ='sal',
                             use_infidelity=False, 
                             use_max_sensitivity=False, 
                             use_sparseness= False, 
                             use_road = False):
    model.zero_grad()
    model.cpu()
    if(xai_method == 'sal'):
        sal = Saliency(model)
    elif (xai_method == 'ig'):
        ig = IntegratedGradients(model)
    else:
        print("Error (interpretability): Please make sure xai_method is either sal or ig")
        return

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
            normalise = False
        )

    if(use_sparseness):
        sparseness = quantus.Sparseness(
            normalise = False,
        )

    if(use_road):
        road = quantus.ROAD(
            noise=0.01,
            perturb_func=quantus.perturb_func.noisy_linear_imputation,
            percentages=list(range(1, 100, 1)),
            normalise = False
        )

    sal_results = {
        'infidelity': [],
        'max_sensitivity': [],
        'sparseness': []
    }

    if(use_road):
        road_results = {

        }
        for i in range(1, 50, 2):
            road_results[i] = []

    for images, _ in dataloader:
        images = images.cpu()
        # labels = labels.cpu()

        with torch.no_grad():  # No need to compute gradients during testing
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
        preds = preds.cpu()
        # Compute attributions
        if(xai_method == 'sal'):      
            attributions = sal.attribute(
                images, target = preds
            ).sum(axis=1).cpu().numpy()
        elif(xai_method == 'ig'):
            attributions = ig.attribute(
            images, baselines=torch.zeros_like(images), target=preds
        ).cpu().numpy()
        attributions = quantus.functions.normalise_func.normalise_by_average_second_moment_estimate(attributions)
        images, preds = images.numpy(), preds.numpy()

        if(use_infidelity):
            sal_results['infidelity'].extend(
                infidelity(model=model,
                        x_batch=images,
                        y_batch=preds,
                        a_batch=attributions)
            )
        if(use_max_sensitivity):
            sal_results['max_sensitivity'].extend(
                max_sensitivity(model=model,
                                x_batch=images,
                                y_batch=preds,
                                a_batch=attributions,
                                explain_func=quantus.explain,
                                explain_func_kwargs={"method": "Saliency", "softmax": False})
            )
        if(use_sparseness):
            sal_results['sparseness'].extend(
                sparseness(model=model,
                        x_batch=images,
                        y_batch=preds,
                        a_batch=attributions)
            )

        if(use_road):
            road_dict = road(model=model,
                        x_batch=images,
                        y_batch=preds,
                        a_batch=attributions,
                        softmax = False)
            for i in range(1, 50, 2):
                road_results[i].append(road_dict[i])


    logger.info("NO. OF RESULTS:" + str(len(sal_results['infidelity'])))

    logger.info(f"######## {xai_method} ##########")
    if(use_infidelity):
        logger.info(f"Mean infidelity {xai_method}: {np.mean(sal_results['infidelity']):.4f}")
        logger.info(f"Median infidelity {xai_method}: {np.median(sal_results['infidelity']):.4f}")

    if(use_max_sensitivity):
        logger.info(f"Mean max-sensitivity {xai_method}: {np.mean(sal_results['max_sensitivity']):.4f}")
        logger.info(f"Median max-sensitivity {xai_method}: {np.median(sal_results['max_sensitivity']):.4f}")

    if(use_sparseness):
        logger.info(f"Mean sparseness {xai_method}: {np.mean(sal_results['sparseness']):.4f}")
        logger.info(f"Median sparseness {xai_method}: {np.median(sal_results['sparseness']):.4f}")

    if(use_road):
        road_means = []
        for i in range(1, 50, 2):
            road_means.append(np.mean(np.array(road_results[i])))
        logger.info(f"Mean ROAD AUC {xai_method}: {auc(range(1, 50, 2),road_means):.4f}")