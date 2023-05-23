from torch.autograd import grad
from torch.utils.data import DataLoader, Dataset
import torch
import copy
from torch import nn

def calc_grad(dataset, model, criterion, gpu=-1, quick=False, create_graph=False):
    """Calculates the gradient z. One calc_grad should be computed for each
    training sample.
    Arguments:
        z: torch tensor, training data points
            e.g. an image sample (batch_size, 3, 256, 256)
        t: torch tensor, training data labels
        model: torch NN, model used to evaluate the dataset
        gpu: int, device id to use for GPU, -1 for CPU
    Returns:
        calc_grad: list of torch tensor, containing the gradients
            from model parameters to loss"""
    data_loader = DataLoader(dataset, batch_size=16, shuffle=False)
    params = [ p for p in model.parameters() if p.requires_grad]
    model.eval()
    gradient = None
    # initialize
    for z, t in data_loader:
        if gpu >= 0:
            z, t = z.cuda(), t.cuda()
        y = model(z)
        loss = criterion(y, t)
        temp_grad = grad(loss, params, create_graph=create_graph)
        # print(temp_grad[-1])
        if quick:
            gradient = [g for g in temp_grad]
        elif gradient is None:
            gradient = [g / len(data_loader) for g in temp_grad]
        else:
            gradient = [(g + g_temp / len(data_loader)) for g, g_temp in zip(gradient, temp_grad)]

    return gradient

def hvp(dataset, model, v, criterion, gpu=-1):
    """Multiply the Hessians of y and w by v.
    Uses a backprop-like approach to compute the product between the Hessian
    and another vector efficiently, which even works for large Hessians.
    Example: if: y = 0.5 * w^T A x then hvp(y, w, v) returns and expression
    which evaluates to the same values as (A + A.t) v.
    Arguments:
        y: scalar/tensor, for example the output of the loss function
        w: list of torch tensors, tensors over which the Hessian
            should be constructed
        v: list of torch tensors, same shape as w,
            will be multiplied with the Hessian
    Returns:
        return_grads: list of torch tensors, contains product of Hessian and v.
    Raises:
        ValueError: `y` and `w` have a different length."""
    w = [ p for p in model.parameters() if p.requires_grad ]
    if len(w) != len(v):
        raise(ValueError("w and v must have the same length."))

    # First backprop
    first_grads = calc_grad(dataset, model, criterion, gpu, quick=False, create_graph=True)

    # Elementwise products
    elemwise_products = 0
    for grad_elem, v_elem in zip(first_grads, v):
        elemwise_products += torch.sum(grad_elem * v_elem)

    # Second backprop
    return_grads = grad(elemwise_products, w, create_graph=False)

    return return_grads

def sum_grad(grad_list):
    result = grad_list[0]
    for grad in grad_list[1:]:
        for p_result, p_grad in zip(result, grad):
            p_result += p_grad
    return list(result)\

def cut_grad(w_local,w_g,device):
    grad = copy.deepcopy(w_local)
    for k in w_local.keys():
        grad[k] =w_g[k].to(device)-w_local[k].to(device)
    return grad


class calculate_gradient(object):
    def __init__(self, batch_size=16, dataset=None, criterion=None):

        self.loss_func = nn.CrossEntropyLoss() if criterion is None else criterion
        self.selected_clients = []
        self.local_bs = batch_size
        self.ldr_train = DataLoader(dataset, batch_size=self.local_bs, shuffle=True)


    def calcluate(self,net,device):
        net.eval()

        loss_all = 0
        for batch_idx, (images, labels) in enumerate(self.ldr_train):

            images, labels = images.to(device), labels.to(device)

            log_probs = net(images)
            # loss = self.loss_func(log_probs, labels)
            loss_function = torch.nn.CrossEntropyLoss()
            loss = loss_function(log_probs, labels) / len(self.ldr_train)
            loss.backward()

        grad_test = []
        for g in net.parameters():
            grad_test.append(g.grad)

        return grad_test


def DIG_FL(w_local, w_g, net,validation_dataset,device):

    contirbution_epoch = []
    cal_gradient_temp = calculate_gradient(dataset=validation_dataset)
    grad_test = cal_gradient_temp.calcluate(net=copy.deepcopy(net.to(device)),device=device)
    for i in range(len(w_local)):
        temp = cut_grad(w_local[i], w_g, device)
        net.load_state_dict(temp)
        grad_client=(list(net.parameters()))
        product = 0
        for (g,v) in zip(grad_client,grad_test):
            product += torch.sum(torch.mul(g,v))
        contirbution_epoch.append(max(product.cpu().item(),0))

    return contirbution_epoch
