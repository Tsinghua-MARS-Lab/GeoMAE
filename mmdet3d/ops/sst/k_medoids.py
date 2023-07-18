from typing import Tuple

import torch


def select_initial_medoids(batched_distance_matrices: torch.Tensor, num_clusters: int) -> torch.Tensor:
    normalized_distances = batched_distance_matrices / (torch.sum(batched_distance_matrices, dim=-1, keepdim=True) + 1e-8)
    initial_scores = normalized_distances.sum(dim=1)
    _, indices = initial_scores.topk(num_clusters, largest=False, sorted=False)
    return indices

def assign_samples_to_medoids(
    batched_distance_matrices: torch.Tensor,
    medoids_indices: torch.Tensor,
) -> Tuple[torch.Tensor, float]:
    batch_size, num_samples, _ = batched_distance_matrices.shape
    _, num_clusters = medoids_indices.shape
    dist_samples_to_medoids = batched_distance_matrices.gather(
        2,
        medoids_indices[:, None, :].expand(batch_size, num_samples, num_clusters)
    )
    min_dist, cluster_assignment = dist_samples_to_medoids.min(dim=2)
    min_dist = min_dist.sum(dim=-1)
    return cluster_assignment, min_dist


def update_medoids(
    medoids_indices: torch.Tensor,
    batched_distance_matrices: torch.Tensor,
    num_clusters: int,
    cluster_assignment: torch.Tensor,
):
    batch_size, num_samples, _ = batched_distance_matrices.shape

    for i in range(num_clusters):
        out_of_cluster_mask = cluster_assignment != i

        cluster_distance_matrices = batched_distance_matrices.clone()
        cluster_distance_matrices[out_of_cluster_mask] = 0
        cluster_distance_matrices[out_of_cluster_mask[:, None, :].expand(batch_size, num_samples, num_samples)] = 0
        dist_sum = torch.sum(cluster_distance_matrices, dim=-1)
        dist_sum[dist_sum == 0] = 1000000
        new_medoids_indices = torch.argmin(dist_sum, dim=-1)
        medoids_indices[:, i] = new_medoids_indices

def k_medoids(
    batched_distance_matrices: torch.Tensor,
    num_clusters: int,
    max_iter: int = 3,
    padding_idx: int = 0,
) -> torch.Tensor:
    """
    K-medoids clustering algorithm.
    :param batched_distance_matrices: batched distance matrices. [batch_size, num_samples, num_samples]
    :param num_clusters: number of clusters.
    :param max_iter: maximum number of iterations.
    :param padding_idx: padding index.

    :return: indices of medoids. [batch_size, num_clusters]
    """

    assert batched_distance_matrices.dim() == 3
    assert batched_distance_matrices.shape[1] == batched_distance_matrices.shape[2]
    batch_size, num_samples, _ = batched_distance_matrices.shape
    device = batched_distance_matrices.device

    if num_clusters >= num_samples:
        indices = torch.full((batch_size, num_clusters), padding_idx, dtype=torch.long, device=device)
        indices[:, :num_samples] = torch.arange(num_samples, dtype=torch.long, device=device)
        return indices

    medoids_indices = select_initial_medoids(batched_distance_matrices, num_clusters)
    cluster_assignment, _ = assign_samples_to_medoids(batched_distance_matrices, medoids_indices)

    for _ in range(max_iter):
        update_medoids(medoids_indices, batched_distance_matrices, num_clusters, cluster_assignment)
        cluster_assignment, _ = assign_samples_to_medoids(batched_distance_matrices, medoids_indices)

    return medoids_indices