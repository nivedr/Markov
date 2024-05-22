import numpy as np
import torch
import torch.nn.functional as F
from contextlib import nullcontext, contextmanager, ExitStack
from copy import deepcopy
import pickle


def get_batch(P, order, seq_length, batch_size, generator, extra_args, device='cpu'):
    data = torch.zeros(batch_size, seq_length+1, device=device)
    # if extra_args.initial == 'steady':
    #     alpha = q / (p+q)
    if extra_args.initial == 'uniform':
        alpha = 0.5
    else:
        alpha = 0.5
    # Generate first k bits
    for k in range(order):
        data[:,k] = torch.bernoulli(alpha*torch.ones((batch_size,), device=device), generator=generator)
    for i in range(order, seq_length):
        data[:,i] = get_next_symbols(P, data[:,i-order:i], device)
    x = data[:,:seq_length].to(int)
    y = data[:,1:].to(int)
    #if "cuda" in torch.device(device).type:
    #    # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
    #    x = x.pin_memory().to(device, non_blocking=True)
    #    y = y.pin_memory().to(device, non_blocking=True)
    return x, y

def CE_estimate(P, order, seq_length, batch_size, generator, extra_args, device='cpu'):
    bool_to_int = torch.tensor([2**i for i in range(order)], device=device)
    data = get_batch(P, order, seq_length, batch_size, generator, extra_args, device='cpu')
    CE_est = 0.0
    
    for i in range(order, seq_length):
        slice = data[:,i-order:i]
        idx = torch.sum(torch.mul(data, bool_to_int[None,:]), dim=1)
        M = P.to(device)[idx.to(int)]
        CE_est -= torch.sum(torch.log(M[data[i]]))
    
    return CE_est/batch_size/(seq_length-order)

def get_next_symbols(P, data, device='cpu'):
    order = data.size(dim=1)
    bool_to_int = torch.tensor([2**i for i in range(order)], device=device)
    idx = torch.sum(torch.mul(data, bool_to_int[None,:]), dim=1)
    
    M = P.to(device)[idx.to(int)]
    s = torch.multinomial(M,1).flatten()

    return s


@torch.no_grad()
def eval(model, tokenizer, P, order, sequence_length, model_width, batch_size, generator, extra_args, device='cpu', max_num_batches=24, ctx=nullcontext()):
    assert model.training == False

    loss_list_val, acc_list = [], []

    for _ in range(max_num_batches): 
        x, _ = get_batch(P, order, sequence_length, batch_size, generator, extra_args, device=device)
        x = pad(tokenizer.encode_batch(x), model_width)
        
        y = deepcopy(x[:,1:]).to("cuda")
        x = deepcopy(x[:,:-1]).to("cuda")

        with ctx:
            outputs = model(x, targets=y, get_logits=True)
        val_loss = outputs['loss']
        loss_list_val.append(val_loss)
        acc_list.append((outputs['logits'].argmax(-1) == y).float().mean())

    val_acc = torch.stack(acc_list).mean().item()
    val_loss = torch.stack(loss_list_val).mean().item()
    val_perplexity = 2.71828 ** val_loss

    return val_acc, val_loss, val_perplexity

@torch.no_grad()
def eval_probs(model, tokenizer, P, order, sequence_length, model_width, generator, extra_args, device='cpu', ctx=nullcontext()):
    assert model.training == False

    loss_list_val, acc_list = [], []

    x, _ = get_batch(P, order, sequence_length, 1, generator, extra_args, device=device)
    x = pad(tokenizer.encode_batch(x), model_width)
    y = deepcopy(x[:,1:]).to("cuda")
    x = deepcopy(x[:,:-1]).to("cuda")

    with ctx:
        outputs = model (x, targets=y, get_logits=True)
    val_loss = outputs['loss']
    loss_list_val.append(val_loss)
    acc_list.append((outputs['logits'].argmax(-1) == y).float().mean())

    probs = F.softmax(outputs['logits'], dim=-1)

    xb = x[0]
    print(xb)
    probsb = probs[0, order-1:]
    idx = xb[:-order+1]

    torch.set_printoptions(profile="full")
    print(probsb)

    print(probsb.size())
    print(idx)
    print(idx.size())

    with open('probsb.pickle', 'wb') as handle:
        pickle.dump(probsb.to("cpu"), handle, protocol=pickle.HIGHEST_PROTOCOL)
    exit()
        
    # vec0 = probsb[idx == 0][:,1] # estimated p
    # vec1 = probsb[idx == 1][:,0] # estimated q
    prob_vec = [vec0, vec1]

    val_acc = torch.stack(acc_list).mean().item()
    val_loss = torch.stack(loss_list_val).mean().item()
    val_perplexity = 2.71828 ** val_loss

    return val_acc, val_loss, val_perplexity, prob_vec

@torch.no_grad()
def eval_sparse(model, P, sequence_length, batch_size, device='cpu', max_num_batches=24, ctx=nullcontext(), alpha_th=None, drop_k=None):
    assert model.training == False

    ce_loss_list_val, l1_loss_list_val, acc_list, sparcity_per_layer = [], [], [], []

    for _ in range(max_num_batches): 
        x, y = get_batch(P, sequence_length, batch_size, device=device)
        with ctx:
            outputs = model(x, targets=y, alpha_th=alpha_th, drop_k=drop_k, get_logits=True, get_alphas=True)
        ce_loss_list_val.append(outputs['ce_loss'])
        l1_loss_list_val.append(outputs['l1_loss'])
        acc_list.append((outputs['logits'].argmax(-1) == y).float().mean())
        sparcity_per_layer.append([(alphas < 1e-8).sum().float().cpu().item() / alphas.numel() for alphas in outputs['alphas']])

    val_acc = torch.stack(acc_list).mean().item()
    val_ce_loss = np.mean(ce_loss_list_val)
    val_l1_loss = np.mean(l1_loss_list_val)
    val_perplexity = 2.71828 ** val_ce_loss
    sparcity_per_layer = np.mean(np.array(sparcity_per_layer), axis=0)

    return val_acc, val_ce_loss, val_l1_loss, val_perplexity, sparcity_per_layer


@torch.no_grad()
def eval_sweep_dropk(model, P, sequence_length, batch_size, n_heads, device='cpu', max_num_batches=24, ctx=nullcontext()):
    assert model.training == False

    x_axis, y_axis_pp, y_axis_acc, y_axis_loss = torch.linspace(0.0,0.95,15), [], [], []
    loss_list_val, acc_list = [], []

    for frac in x_axis:
        drop_k = int(sequence_length * frac * n_heads)
        for _ in range(max_num_batches): 
            x, y = get_batch(P, sequence_length, batch_size, device=device)
            with ctx:
                outputs = model(x, targets=y, alpha_th=None, drop_k=drop_k, get_logits=True)
            loss_list_val.append(outputs['ce_loss'])
            acc_list.append((outputs['logits'].argmax(-1) == y).float().mean())

        y_axis_acc.append(torch.stack(acc_list).mean().item())
        y_axis_loss.append(np.mean(loss_list_val))
        y_axis_pp.append(2.71828 ** y_axis_loss[-1])

    return x_axis, y_axis_acc, y_axis_pp, y_axis_loss


@torch.no_grad()
def eval_sweep_alphath(model, P, sequence_length, batch_size, device='cpu', max_num_batches=24, ctx=nullcontext()):
    assert model.training == False

    alpha_ths, y_axis_pp, y_axis_acc, y_axis_loss = [0, 1e-4, 1e-3, 1e-2, 1e-1, 2e-1, 3e-1, 4e-1, 5e-1], [], [], []
    loss_list_val, acc_list, x_axis = [], [], []

    for alpha_th in alpha_ths:
        frac_heads_pruned_list = []
        for _ in range(max_num_batches): 
            x, y = get_batch(P, sequence_length, batch_size, device=device)
            with ctx:
                outputs = model(x, targets=y, alpha_th=alpha_th, drop_k=None, get_logits=True)
            nph, nh = outputs['num_head_pruned_per_layer'], outputs['num_heads_per_layer']
            frac_heads_pruned = np.sum(nph) / np.sum(nh) # fractions of heads removed given alpha_th
            frac_heads_pruned_list.append(frac_heads_pruned)
            loss_list_val.append(outputs['ce_loss'])
            acc_list.append((outputs['logits'].argmax(-1) == y).float().mean())

        x_axis.append(np.mean(frac_heads_pruned_list))
        y_axis_acc.append(torch.stack(acc_list).mean().item())
        y_axis_loss.append(np.mean(loss_list_val))
        y_axis_pp.append(2.71828 ** y_axis_loss[-1])

    return x_axis, y_axis_acc, y_axis_pp, y_axis_loss


def save_checkpoint(distributed_backend, model, opt, scheduler, itr, ckpt_path, **extra_args):

    checkpoint = dict({
        'model': distributed_backend.get_raw_model(model).state_dict(),
        'optimizer': opt.state_dict(),
        'scheduler': scheduler.state_dict(),
        'itr': itr,
    }, **extra_args)

    torch.save(checkpoint, ckpt_path)

@torch.no_grad()
def pad(x, model_width):
    if x.size()[1] > model_width:
        return x[...,:model_width]
    else:
        return torch.nn.functional.pad(x, (0, model_width-x.size()[1], 0, 0))
