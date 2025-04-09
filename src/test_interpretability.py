import torch
import logging
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np

from torchvision import models
from sklearn.metrics import auc
from captum.attr import IntegratedGradients, Saliency, DeepLift #maybe add kernelSHAP?
import quantus
def compute_batched_alignment(image_batch,sal_batch):
    """
    inputs:
        image_batch: tensor of size(batch_size,channel, height, width) representing image
        sal_batch: tensor of size(batch_size,channel, height, width) representing saliency map 
    return:
        alignment: tensor of size(batch_size) representing alignment between image and saliency map
    """
    images_flat = image_batch.view(image_batch.size(0), -1)
    sal_flat = sal_batch.view(sal_batch.size(0), -1)
    alignment = torch.abs(F.cosine_similarity(images_flat, sal_flat, dim=1))
    return alignment

def compute_batched_road(model, images, labels, attributions, road, monotonic_decrease = False):
    """
    inputs:
        model: the model
        images: tensor of size(batch_size,channel, height, width) representing image
        attributions: tensor of size(batch_size,channel, height, width) representing saliency map
        labels: tensor of size(batch_size) representing the labels the explanation was taken with respect to
        road: quantus ROAD object
    return:
        dictionary of ROAD scores with {percentage perturbed : batch average score}
    """
    images = images.cpu().numpy()
    labels = labels.cpu().numpy()
    attributions = attributions.sum(axis=1).cpu().detach().numpy()

    road_dict = road(model=model,
        x_batch=images,
        y_batch=labels,
        a_batch=attributions,
        softmax = False)
    
    curr_min = 1
    if(monotonic_decrease):
        for key in road_dict:
            curr_min = min(road_dict[key], curr_min)
            road_dict[key] = curr_min

    return road_dict

def compute_batched_maxsens(model, images, labels, attributions, max_sensitivity):
    """
    inputs:
        model: the model
        images: tensor of size(batch_size,channel, height, width) representing image
        attributions: tensor of size(batch_size,channel, height, width) representing saliency map
        labels: tensor of size(batch_size) representing the labels the explanation was taken with respect to
        max_sensitivity: quantus Max_Sensitivity object
    return:
        array-like of size(batch_size) representing max-sensitivity values
    """
    images = images.cpu().numpy()
    labels = labels.cpu().numpy()
    attributions = attributions.sum(axis=1).cpu().detach().numpy()
    attributions = quantus.functions.normalise_func.normalise_by_average_second_moment_estimate(attributions)

    max_sensitivity = quantus.MaxSensitivity(nr_samples=100,
        lower_bound=0.1,
        norm_numerator=quantus.norm_func.fro_norm,
        norm_denominator=quantus.norm_func.fro_norm,
        perturb_func=quantus.perturb_func.uniform_noise,
        similarity_func=quantus.similarity_func.difference,
        normalise= False)

    return max_sensitivity(model=model,
        x_batch=images,
        y_batch=labels,
        a_batch=attributions,
        explain_func=quantus.explain,
        explain_func_kwargs={"method": "Saliency", "softmax": False})    

def compute_batched_sparseness(model, images, labels, attributions, sparseness):
    """
    inputs:
        model: the model
        images: tensor of size(batch_size,channel, height, width) representing image
        attributions: tensor of size(batch_size,channel, height, width) representing saliency map
        labels: tensor of size(batch_size) representing the labels the explanation was taken with respect to
        sparseness: quantus Sparseness object
    return:
        array-like of size(batch_size) representing sparseness of the saliency maps
    """
    images = images.cpu().numpy()
    labels = labels.cpu().numpy()
    attributions = attributions.sum(axis=1).cpu().detach().numpy()
    attributions = quantus.functions.normalise_func.normalise_by_average_second_moment_estimate(attributions)


    return sparseness(model=model,
                        x_batch=images,
                        y_batch=labels,
                        a_batch=attributions)

def compute_batched_infidelity(model, images, labels, attributions,infidelity):
    """
    inputs:
        model: the model
        images: tensor of size(batch_size,channel, height, width) representing image
        attributions: tensor of size(batch_size,channel, height, width) representing saliency map
        labels: tensor of size(batch_size) representing the labels the explanation was taken with respect to
        infidelity: quantus infidelity object
    return:
        array-like of size(batch_size) representing infidelity of the saliency maps
    """
    images = images.cpu().numpy()
    labels = labels.cpu().numpy()
    attributions = attributions.sum(axis=1).detach().cpu().numpy()
    attributions = quantus.functions.normalise_func.normalise_by_average_second_moment_estimate(attributions)

    return infidelity(model=model,
                        x_batch=images,
                        y_batch=labels,
                        a_batch=attributions)

def interpretability_metrics(model, road_proxy_model, dataloader, run_id, xai_method ='sal',
                             use_ground_truth = True,
                             use_infidelity=False,
                             use_alignment=False,
                             use_max_sensitivity=False,
                             use_sparseness= False,
                             use_road = False):
    #initialise models and metrics
    model.zero_grad()
    model.cpu()
    road_proxy_model.cpu()
    if(xai_method == 'sal'):
        sal = Saliency(model)
    elif (xai_method == 'ig'):
        ig = IntegratedGradients(model)
    elif (xai_method == 'deeplift'):
        dl = DeepLift(model)
    elif(xai_method == 'random'):
        pass
    else:
        print("Error (interpretability): Please make sure xai_method is either sal or ig")
        return
    
    ROAD_RANGE = range(0,100,5)

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
            percentages=list(ROAD_RANGE),
            normalise = False
        )

    sal_results = {
        'alignment': [],
        'infidelity': [],
        'max_sensitivity': [],
        'sparseness': []
    }

    if(use_road):
        road_results = {

        }
        for i in ROAD_RANGE:
            road_results[i] = []

    # Compute attributions
    for batch_id, (images, labels) in enumerate(dataloader):
        print("Processing batch",batch_id, " out of" ,len(dataloader))
        images = images.cpu()
        labels = labels.cpu()

        if(not use_ground_truth):
            with torch.no_grad():  # No need to compute gradients during testing
                outputs = model(images)
                _, labels = torch.max(outputs, 1)
            labels = labels.cpu()
        
        if(xai_method == 'sal'):
            attributions = sal.attribute(
                images, target = labels
            )
        elif(xai_method == 'ig'):
            attributions = ig.attribute(
            images, baselines=torch.zeros_like(images), target=labels
        )
        elif (xai_method == 'deeplift'):
            attributions = dl.attribute(
            images, target=labels)
        elif(xai_method == 'random'):
            attributions = torch.rand_like(images)

        if(use_infidelity):
            sal_results['infidelity'].extend(
                compute_batched_infidelity(model=model,
                        images=images,
                        labels=labels,
                        attributions=attributions,
                        infidelity=infidelity)
            )
        if(use_alignment):
            sal_results['alignment'].extend(
                compute_batched_alignment(
                    image_batch=images,
                    sal_batch=attributions
                )
            )
        if(use_max_sensitivity):
            sal_results['max_sensitivity'].extend(
                compute_batched_maxsens(model=model,
                                images=images,
                                labels=labels,
                                attributions=attributions,
                                max_sensitivity=max_sensitivity)    
            )
        if(use_sparseness):
            sal_results['sparseness'].extend(
                compute_batched_sparseness(model=model,
                        images=images,
                        labels=labels,
                        attributions=attributions,
                        sparseness=sparseness)
            )

        if(use_road):
            road_dict = compute_batched_road(model=road_proxy_model,
                        images=images,
                        labels=labels,
                        attributions=attributions,
                        road=road,
                        monotonic_decrease=False)
            for i in ROAD_RANGE:
                road_results[i].append(road_dict[i])

    #LOG RESULTS
    logger = logging.getLogger(run_id)
    logger.info(f"######## {xai_method} ##########")
    if(use_infidelity):
        logger.info(f"Mean infidelity {xai_method}: {np.mean(sal_results['infidelity']):.4f}")
        logger.info(f"Median infidelity {xai_method}: {np.median(sal_results['infidelity']):.4f}")

    if(use_alignment):
        logger.info(f"Mean alignment {xai_method}: {np.mean(sal_results['alignment']):.4f}")
        logger.info(f"Median alignment {xai_method}: {np.median(sal_results['alignment']):.4f}")

    if(use_max_sensitivity):
        logger.info(f"Mean max-sensitivity {xai_method}: {np.mean(sal_results['max_sensitivity']):.4f}")
        logger.info(f"Median max-sensitivity {xai_method}: {np.median(sal_results['max_sensitivity']):.4f}")

    if(use_sparseness):
        logger.info(f"Mean sparseness {xai_method}: {np.mean(sal_results['sparseness']):.4f}")
        logger.info(f"Median sparseness {xai_method}: {np.median(sal_results['sparseness']):.4f}")

    if(use_road):
        road_means = []
        for i in ROAD_RANGE:
            road_means.append(np.mean(np.array(road_results[i])))
        logger.info(f"Mean ROAD AUC {xai_method}: {auc(ROAD_RANGE,road_means):.4f}")