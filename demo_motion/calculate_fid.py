
# run with python -m demo_motion.calculate_fid

import os

import numpy as np
import torch
from scipy import linalg

from .motion_metrics_encoder import MotionEncoder


# from action2motion
def calculate_fid(statistics_1, statistics_2):
    return calculate_frechet_distance(statistics_1[0], statistics_1[1],
                                      statistics_2[0], statistics_2[1])


def calculate_activation_statistics(activations):
    activations = activations.cpu().detach().numpy()
    # activations = activations.cpu().numpy()
    mu = np.mean(activations, axis=0)
    sigma = np.cov(activations, rowvar=False)
    return mu, sigma


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)


#adapted from action2motion
def calculate_diversity(activations):
    diversity_times = 200
    num_motions = len(activations)

    diversity = 0

    first_indices = np.random.randint(0, num_motions, diversity_times)
    second_indices = np.random.randint(0, num_motions, diversity_times)
    for first_idx, second_idx in zip(first_indices, second_indices):
        diversity += torch.dist(activations[first_idx, :],
                                activations[second_idx, :])
    diversity /= diversity_times
    return diversity


def test():

    checkpoint = "./runs/motion/motion-encoder-267.pt"
    dsg_path = "/move/u/yifengj/TML_diffusion_AMASS/data/may6/dsg200"
    trust_path = "/move/u/yifengj/TML_diffusion_AMASS/data/may6/trust50"
    gt_path = "/move/u/yifengj/TML_diffusion_AMASS/data/may6/AMASS_test_aggregated_sliced"

    model = MotionEncoder(checkpoint, predict_contact=False, use_masks=False)
    model.eval()
    assert model.normalizer is not None

    list_of_paths = [dsg_path, trust_path, gt_path]
    list_of_motions = []
    for path in list_of_paths:
        motions = []
        # list all files in the directory
        for file in os.listdir(path):
            if file.endswith(".pt"):
                data = torch.load(os.path.join(path, file))
                motions.append(data)
        motions = torch.stack(motions, dim=0)
        print("motions.shape", motions.shape)
        if motions.shape[-1] > 135:
            motions = motions[:, :, 4:]
            print("motions.shape new", motions.shape)
        list_of_motions.append(motions)             # list of (num_motions, 60, 135)
    
    # calculate FID and diversity
    activations_dsg = model.encode(list_of_motions[0].to(model.accelerator.device))
    activations_trust = model.encode(list_of_motions[1].to(model.accelerator.device))
    activations_gt = model.encode(list_of_motions[2].to(model.accelerator.device))

    mu_dsg, sigma_dsg = calculate_activation_statistics(activations_dsg)
    mu_trust, sigma_trust = calculate_activation_statistics(activations_trust)
    mu_gt, sigma_gt = calculate_activation_statistics(activations_gt)

    fid_dsg_gt = calculate_fid((mu_dsg, sigma_dsg), (mu_gt, sigma_gt))
    fid_trust_gt = calculate_fid((mu_trust, sigma_trust), (mu_gt, sigma_gt))
    print("FID DSG vs GT:", fid_dsg_gt)
    print("FID Trust vs GT:", fid_trust_gt)

    diversity_dsg = calculate_diversity(activations_dsg)
    diversity_trust = calculate_diversity(activations_trust)

    print("Diversity DSG:", diversity_dsg)
    print("Diversity Trust:", diversity_trust)
    print("Diversity GT:", calculate_diversity(activations_gt))


    # train_eval_dataset = MotionDataset(
    #     data_path=opt.data_dir,
    #     backup_path=None,
    #     train=False,
    #     force_reload=True,
    #     horizon_trim=54,          # truncate to horizon if not already
    # )
    # ll = len(train_eval_dataset)
    # print("len(train_dataset)", ll)

    # # shuffle
    # indices = np.random.permutation(ll)
    # train_eval_dataset = torch.utils.data.Subset(train_eval_dataset, indices)

    # train_eval_dataset_last_batch = train_eval_dataset[ll-num_samples:ll].to(model.accelerator.device)
    # print("len(train_dataset_last_batch)", len(train_eval_dataset_last_batch))

    # activations = model.encode(train_eval_dataset_last_batch)
    # print("activations.shape", activations.shape)

    # train_eval_dataset_2nd_last_batch = train_eval_dataset[ll-2*num_samples:ll-num_samples].to(model.accelerator.device)      # (num_samples, 60, 135)
    # train_eval_dataset_2nd_last_batch = 0.8 * train_eval_dataset_2nd_last_batch + 0.2 * torch.randn_like(train_eval_dataset_2nd_last_batch)

    # print("len(train_dataset_2nd_last_batch)", len(train_eval_dataset_2nd_last_batch))

    # activations_2nd = model.encode(train_eval_dataset_2nd_last_batch)
    # print("activations_2nd.shape", activations_2nd.shape)

    # mu1, sigma1 = calculate_activation_statistics(activations)
    # mu2, sigma2 = calculate_activation_statistics(activations_2nd)
    # fid = calculate_fid((mu1, sigma1), (mu2, sigma2))
    # print("FID:", fid)

    # diversity = calculate_diversity(activations)
    # print("Diversity:", diversity)


if __name__ == "__main__":

    test()