import torch
from transformers import AutoModelForCausalLM, AutoTokenizer



def compute_hvp(model, layer_name, vec, *loss_args):
    """
    Compute the Hessian-vector product of a model with respect to a loss function at a given point.

    :param model: model to compute the Hessian-vector product for, nn.Module
    :param loss_fn: loss function to compute the Hessian-vector product for, callable and differentiable, output is a scalar
    :param vec: vector to compute the Hessian-vector product with, torch.Tensor, size is the number of parameters in the model
    :param loss_args: arguments to pass to the loss function

    e.g. Hv = compute_hvp(model, loss_fn, vec, teacher_model, input)

    :return: Hessian-vector product, torch.Tensor, size is the number of parameters in the model, w
    """
    loss = kl_loss_fn(model, layer_name, *loss_args)

    params = [p for p in model.parameters() if p.requires_grad]
    gradients = torch.autograd.grad(loss, params, create_graph=True)

    # compute gradient * vector
    flat_gradients = torch.cat([g.view(-1) for g in gradients])
    gv = torch.dot(flat_gradients, vec)
    
    # compute Hessian-vector product
    hvp = torch.autograd.grad(gv, params)

    # flatten the Hessian-vector product by row first
    hvp = torch.cat([h.view(-1) for h in hvp])
    print(f"hvp of layer {layer_name}: shape is {hvp.shape}")
    return hvp

def kl_loss_fn(model, target_layer_name, teacher_model, input):
    """
    Compute the Kullback-Leibler divergence between the model and the teacher model.

    To get a single layer's Hessian-Vector Product, only the target layer requires gradient, the rest of the model is frozen.

    :param model: model to compute the Kullback-Leibler divergence for, nn.Module
    :param teacher_model: teacher model to compute the Kullback-Leibler divergence for, nn.Module
    :param input: input to the model, dict

    :return: Kullback-Leibler divergence, torch.Tensor, size is 1
    """

    # Compute the whole model weights 
    if target_layer_name is "":
        pass
    else:
        for name, param in model.named_parameters():
            param.requires_grad = False

        
        for name, param in model.named_parameters():
            if name == target_layer_name:
                param.requires_grad = True
                print(f"Unfreezing: {name}")


    output = model(**input)
    with torch.no_grad():
        teacher_output = teacher_model(**input)
    return torch.nn.KLDivLoss()(output.logits, teacher_output.logits)


### For testing purposes
### add `attn_implementation="eager"` to the model initialization to ensure derivative computation

if __name__ == '__main__':

    model = AutoModelForCausalLM.from_pretrained("gpt2", attn_implementation="eager")
    total_params = 0
    for name, param in model.named_parameters():
        print(f"Layer: {name} | Size: {param.size()} | Number of parameters: {param.numel()}")
        total_params += param.numel()
    print(f"Total number of parameters: {total_params}")

    teacher_model = AutoModelForCausalLM.from_pretrained("gpt2", attn_implementation="eager")
    device = torch.device("cuda:0")
    model.to(device)
    teacher_model.to(device)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    input = tokenizer("Hello, my dog is cute", return_tensors="pt").to(device)
    input["labels"] = input["input_ids"].clone()

    target_layer = "transformer.h.0.mlp.c_fc.weight"

    vec = torch.randn(torch.numel(torch.cat([param.view(-1) for name, param in model.named_parameters() if name == target_layer and param.requires_grad]))).to(device)
    hvp = compute_hvp(model, target_layer, vec, teacher_model, input)
    print(hvp)