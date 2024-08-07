import os
import sys
import numpy as np
import torch
import inspect
import json
import copy
import argparse
import random
import wandb
import sys
sys.path.append('tokenizer')
import train_tokenizer
import Tokenizer
import BPE

import config
from models.base import AddBeta
from models.utils import get_model
from optim.base import train_base
from optim.sparse import train_sparse
from optim.utils import get_batch, CE_estimate
import distributed


def get_args():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--config_format', default='markov', choices=config.registered_formats())
    parser.add_argument('--tokenizer', default='BPE', choices=['Character', 'BPE', 'LZW'])
    parser.add_argument('--alphabet_size', type=int, default=2)
    parser.add_argument('--max_dict_size', type=int, default=10)
    parser.add_argument('--dataset_size', type=int, default=10000)
    parser.add_argument('--transition', default='switching', choices=['random', 'switching', 'interpolation'])
    parser.add_argument('--interpolation', default=0.1)
    
    args, rem_args = parser.parse_known_args()

    return config.parse_args_with_format(format=args.config_format, base_parser=parser, args=rem_args, namespace=args)


def get_exp_name(args):
    """ Returns the name of the experiment, used for saving models and wandb. """
    exp_name = f"{args.model}_lr{args.lr}_bs{args.batch_size}x{args.acc_steps}_{args.world_size}nodes"
    if args.wandb_run_prefix != 'none':
        exp_name = args.wandb_run_prefix + '_' + exp_name
    if 'sparse' in args.model:
        exp_name += f"_lmd{args.lmbda}"
    exp_name += f"_seed={args.seed}"
    return exp_name


def main(args): 
    # Markov transition probabilities
    #I = torch.eye(args.vocab_size)
    #P = torch.zeros(args.vocab_size, args.vocab_size)
    #P[:,0] = I[:,-1]
    #P[:,1:] = I[:,:-1]
    # p = args.p # 0... -> 1
    # q = args.q # 1... -> 0
    
    tokenizer = args.tokenizer
    order = int(args.order)
    delta = args.interpolation
    alphabet_size = int(args.alphabet_size)
    generator = torch.Generator(device=args.device)
    generator.seed()
    cpu_generator = torch.Generator(device='cpu')
    cpu_generator.seed()

    if args.transition == 'random':
        P = torch.rand([alphabet_size**order,alphabet_size], generator=cpu_generator)
        sum_P = torch.sum(P, dim=1)
        P = torch.transpose(torch.div(torch.transpose(P,0,1), sum_P),0,1)
        # P = torch.cat((P,1-P),dim=1)
    elif args.transition == 'switching':
        p = args.p
        q = args.q
        P = torch.Tensor([[1-p, p],[q, 1-q]]) # [ P(.| ..., 0) ; P(.| ...,1) ]
        P = P.repeat(2**(order-1),1)
    elif args.transition == 'interpolation':
        p = args.p
        q = args.q

        # transition 1 (switching): P
        P = torch.Tensor([[1-p, p],[q, 1-q]]) # [ P(.| ..., 0) ; P(.| ...,1) ]
        P = P.repeat(2**(order-1),1)

        # transition 2 (random): Q
        Q = torch.rand([2**order,1], generator=cpu_generator)
        Q = torch.cat((Q,1-Q),dim=1)

        # Real transition = (1-delta) P + delta Q
        P = (1 - delta)*P + delta*Q
    args.P = P

    torch.backends.cuda.matmul.allow_tf32 = True # allows us to make sure we're able to use tensorfloat32 during training
    torch.backends.cudnn.allow_tf32 = True

    distributed_backend = distributed.make_backend_from_args(args)
    args = distributed_backend.get_adjusted_args_for_process(args)

    args.device = torch.device(args.device)
    device_type = "cuda" if "cuda" in str(args.device) else "cpu"
    if device_type == "cuda":
        torch.cuda.set_device(args.device)

    #torch.manual_seed(args.seed)
    #random.seed(args.seed)
    #np.random.seed(args.seed)
    
    print(f"Loading dataset '{args.dataset}'")

    max_dict_size=args.max_dict_size
    dataset_size=args.dataset_size
    tokenizer_model = train_tokenizer.train_tokenizer(tokenizer, max_dict_size, P, order, alphabet_size, generator=cpu_generator, dataset_size=dataset_size, extra_args=args)

    est = CE_estimate(P, order, alphabet_size, args.sequence_length, 50, cpu_generator, extra_args=args, device='cpu')
    print(f"Cross entropy estimate of the Markov chain is: {est}")
    
    tok_len = []
    for i in range(10):
        x, _ = get_batch(P, order, alphabet_size=alphabet_size, seq_length=args.sequence_length, batch_size=1, generator=generator, extra_args=args, device=device_type)
        x = tokenizer_model.encode_batch(x)
        tok_len.append(x.size()[1])

    # char_len = args.sequence_length
    # args.sequence_length = int(np.mean(tok_len))
    # print(args.sequence_length)

    # args.vocab_size = args.max_dict_size
    
    # char_len = args.sequence_length
    # # tok_len = args.sequence_length
    # args.sequence_length = int(np.mean(tok_len))
    # print(args.sequence_length)

    print(f"Transformer width is: {int(np.mean(tok_len))}")
    print(f"Seq_l to width Ratio is: {int(np.mean(tok_len))/args.sequence_length}")

    args.vocab_size = args.max_dict_size
    model = get_model(args).to(args.device) # todo: take care of initializing the model if args.use_pretrained != 'none'

    print(model)
    model = distributed_backend.transform_model(model)
    
    group_specs = distributed_backend.get_raw_model(model).get_parameter_group_specs()
    param_name_mapping = {p_name: p for p_name, p in model.named_parameters()}
    optimized_params_cnt = 0
    for g in group_specs:
        params = []
        for p_name in g["params"]:
            translated_p_names = distributed_backend.translate_model_parameter_name_for_node(p_name)
            params += [param_name_mapping[p_name] for p_name in translated_p_names]
        g["params"] = params
        optimized_params_cnt += sum([p.numel() for p in g["params"]])
    print("number of optimized parameters: %.2fM" % (optimized_params_cnt/1e6,))
    if args.opt == 'adamw':
        use_fused = (device_type == 'cuda') and ('fused' in inspect.signature(torch.optim.AdamW).parameters)
        print(f"using fused AdamW: {use_fused}")
        extra_args = dict(fused=True) if use_fused else dict()
        opt = torch.optim.AdamW(group_specs, lr=args.lr, betas=(args.beta1, args.beta2),
                                weight_decay=args.weight_decay, **extra_args)
    else:
        opt = torch.optim.SGD(group_specs, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    
    if args.scheduler != 'none':
        if args.scheduler in ['cos', 'linear']:
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=opt, max_lr=args.lr, total_steps=args.iterations, 
                                                            pct_start=args.warmup_percent, anneal_strategy=args.scheduler, 
                                                            cycle_momentum=False, div_factor=1e2, final_div_factor=.05)
        else:
            raise NotImplementedError(f"Unknown scheduler type: {args.scheduler}.")
    else:
        scheduler = None

    args.world_size = distributed_backend.get_world_size()
    exp_name = get_exp_name(args)
    if distributed_backend.is_master_process() and args.wandb:
        params_copy = copy.deepcopy(vars(args))
        del params_copy['device']
        wandb.init(project=args.wandb_project, name=exp_name, config=params_copy)
    
    ckpt_path = os.path.join(args.results_base_folder, args.dataset, args.model, exp_name)
    if not os.path.exists(ckpt_path):
        if distributed_backend.is_master_process():
            os.makedirs(ckpt_path)
    elif os.path.isfile(os.path.join(ckpt_path, "summary.json")): # the experiment was already completed
        print(f"Already found experiment '{ckpt_path}'.\nSkipping.")
        sys.exit(0)

    if args.model == 'base': # all train functions have the same interface
        train = train_base
    elif 'sparse' in args.model:
        train = train_sparse
    else:
        raise NotImplementedError(f"No training method implemented for model type '{args.model}'.")

    print(f"\nTraining model={args.model} \n{vars(args)}\n")
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    print("Training transformer...")
    stats = train(model, tokenizer_model, opt, P, order, alphabet_size, scheduler, args.iterations, args.acc_steps, args.batch_size, args.sequence_length, int(np.mean(tok_len)), generator,
                  eval_freq=args.eval_freq, 
                  distributed_backend=distributed_backend,
                  ckpt_path=f"{ckpt_path}/ckpt.pt", extra_args=args)

    torch.save(model.state_dict(), 'model.pt')

    args.device = None
    args.dtype = None
    stats['args'] = vars(args)
    if distributed_backend.is_master_process():
        with open(f"{ckpt_path}/summary.json", "w") as fs:
            json.dump(stats, fs)
    distributed_backend.finalize()


if __name__ == "__main__":
    args = get_args()
    main(args)
