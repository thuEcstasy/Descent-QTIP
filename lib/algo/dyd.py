import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
import numpy as np
from scipy.stats import norm
import glog
def decompose_matrix(matrix, iters=24):
    """
    Decompose a matrix into D_1 (n,), Y (n, m), D_2 (m,)
    such that Y = D_1 @ matrix @ D_2 is balanced (rows and columns normalized).
    
    :param matrix: matrix to decompose, shape: (n, m)
    :param iters: number of iterations
    
    :return: D_1, Y, D_2
    """
    n, m = matrix.shape
    # Initialize D_1 and D_2 to ones (diagonal = 1), avoids instability
    print("matrix device: ", matrix.device, flush=True)
    D_1 = torch.ones(n, device=matrix.device)
    D_2 = torch.ones(m, device=matrix.device)

    # random permutation of rows and columns of matrix
    # store the permutation for later use
    # row_permutation = torch.randperm(n)
    # col_permutation = torch.randperm(m)
    # matrix = matrix[row_permutation][:, col_permutation]

    Y = matrix
    for _ in range(iters):

        rowwise_max = Y.abs().max(dim=1, keepdim=True).values
        D_1_update = rowwise_max
        Y = Y / D_1_update
        D_1 *= D_1_update.squeeze()

        D_2_update = Y.std(dim=0, keepdim=True)
        Y = Y / D_2_update
        D_2 *= D_2_update.squeeze()

    error = D_1[:, None] * Y * D_2[None, :] - matrix

    # print("Reconstruction error:", torch.norm(error).item(),flush=True)
    # print the maximal k values of Y
    # print("Maximal k values of Y:", Y.abs().flatten().topk(100).values,flush=True)

    return D_1, Y, D_2
    # return D_1, Y, D_2, row_permutation, col_permutation

def process_attention_and_linear_layers(model):
    decomposed_layers = {}

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and name != "lm_head":
            print(f"Processing Linear layer: {name}")
            weight = module.weight.data

            # calculate mean and std of the weight matrix
            maximal = weight.abs().max().item()
            mean = weight.mean().item()
            std = weight.std().item()
            print(f"Mean: {mean}, Std: {std}", f"Maximal: {maximal}")

            # DYD decomposition
            D_1, Y, D_2 = decompose_matrix(weight)

            # compare distribution of normalized weight and random gaussian
            print(f"Decomposed mean: {Y.mean().item()}, Decomposed std: {Y.std().item()}, Decomposed maximal: {Y.abs().max().item()}")


def main(hf_model_path, rank=128):

    model = AutoModelForCausalLM.from_pretrained(hf_model_path, torch_dtype=torch.float32)
    decomposed_layers = process_attention_and_linear_layers(model)
    print(f"Total decomposed layers: {len(decomposed_layers)}")

if __name__ == "__main__":
    hf_model_path = "meta-llama/Meta-Llama-3-8B"
    main(hf_model_path)