"""
Utilities for fine tuning
"""
import copy
import math
from contextlib import contextmanager
from operator import attrgetter

import glog
import torch
from torch import multiprocessing as mp
from torch import nn
from transformers import AutoModelForCausalLM

from lib import codebook, utils
from lib.linear import QuantizedLinear

from . import ldlq, dyd


@contextmanager
def use_tf32():
    fp32_matmul_precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision('high')
    yield
    torch.set_float32_matmul_precision(fp32_matmul_precision)


def finetune_decoder_layer(layer, name, device, train_dl, valid_dl, orig_dtype,
                           args):
    with use_tf32():
        layer = layer.to(device)

        source = next(iter(train_dl))[0]
        position_ids = torch.arange(source.shape[1], device=device).unsqueeze(0)
        # manifest tensor parallel attributes in layer
        output = layer(source.to(device),
                       position_ids=position_ids)[0]
        
        best_sd = {k: v.cpu() for k, v in layer.state_dict().items()}
        utils.clean()

        optim = torch.optim.Adam(layer.parameters(), lr=args.ft_lr)
        best_loss = utils.calculate_mse_loss(layer, valid_dl, device)
        glog.info(f'layer {name} initial loss {best_loss}')
        scaler = torch.cuda.amp.GradScaler(enabled=(orig_dtype==torch.float16))
        worse_ct = 0

        for epoch in range(args.ft_epochs):
            for bidx, (source, targets) in enumerate(train_dl):
                targets = targets.to(device, non_blocking=True)
                with torch.autocast(device_type='cuda',
                                    dtype=orig_dtype,
                                    enabled=True):
                    output = layer(source.to(device),
                                   position_ids=position_ids)[0]
                    loss = nn.MSELoss()(output, targets)
                scaler.scale(loss).backward()
                if bidx % args.ft_update_freq == args.ft_update_freq - 1 or bidx == len(
                        train_dl) - 1:
                    scaler.step(optim)
                    scaler.update()
                    optim.zero_grad()

            if epoch % args.ft_valid_freq == (args.ft_valid_freq - 1):
                test_loss = utils.calculate_mse_loss(layer, valid_dl, device)
                if test_loss < best_loss:
                    glog.info(
                        f'layer {name} @ epoch {epoch} new loss {test_loss} old loss {best_loss} BETTER'
                    )
                    best_loss = test_loss
                    best_sd = {k: v.cpu() for k, v in layer.state_dict().items()}
                    utils.clean()
                    worse_ct = 0
                else:
                    glog.info(
                        f'layer {name} @ epoch {epoch} new loss {test_loss} old loss {best_loss} WORSE'
                    )
                    worse_ct += 1
                    if worse_ct >= args.ft_early_stop:
                        break

    del optim, train_dl, valid_dl

    layer = layer.cpu()
    layer.load_state_dict(best_sd)
    utils.clean()


def quantize_finetune_decoder_layer(mixed_layer, quant_order, idx, cb, args,
                                    device, pre_orig_emb, orig_emb):
    torch.manual_seed(idx)
    torch.set_num_threads(args.num_cpu_threads)
    torch.set_grad_enabled(False)

    dtype_ = torch.float64 if args.use_fp64 else torch.float32
    orig_dtype = None
    for p in mixed_layer.parameters():
        orig_dtype = p.dtype
        break
    mixed_layer = mixed_layer.float()

    # for tensor-wise fine-tuning
    # train_dl, valid_dl = utils.split_data(pre_orig_emb, orig_emb, args)

    has_kernel = utils.has_kernel(args.decode_mode, args.L, args.K, args.V,
                                  args.tlut_bits, args.td_x, args.td_y)

    for quant_i, (linear_attr, name, in_hess_name, out_hess_name,
                  rcp) in enumerate(quant_order):
        utils.clean()

        # For each weight matrix of student model, compute DYD Decomposition

        orig_linear = attrgetter(linear_attr)(mixed_layer)
        D1, Y, D2 = dyd.decompose_matrix(orig_linear.weight.to(dtype_))

        glog.info(f'computed {orig_linear} DYD Decomposition')    

        cb = cb.to(device).to(orig_dtype)
        orig_linear = attrgetter(linear_attr)(mixed_layer)
        W = orig_linear.weight.to(dtype_)
        del orig_linear
        (m, n) = W.shape
        
        # disable kernel
        has_kernel = utils.has_kernel(args.decode_mode, args.L, args.K, args.V,
                                  args.tlut_bits, args.td_x, args.td_y)

        in_hess_path = f'{args.in_hess_path}/{idx}_{in_hess_name}.pt'
        H_data = torch.load(in_hess_path, map_location=torch.device('cpu'))
        HR = utils.flat_to_sym(H_data['flatH'], H_data['n'])
        if 'mu' in H_data:
            mu = H_data['mu']
            HR += mu[None, :] * mu[:, None]
            del mu
        del H_data
        HR = utils.regularize_H(HR, args.sigma_reg).to(device)

        # calculate an estimation of D, where x^T D x is approximately x^T H x
        DTilde = torch.diagonal(HR)
        DTilde_minus_half = DTilde.pow(-0.5).to(device)
        # by power iteration, find the largest eigenvalue of D^{-1/2} H D^{-1/2}
        v = torch.randn(n, device=device).unsqueeze(1)
        HTilde = DTilde_minus_half[:, None] * HR * DTilde_minus_half[None, :]
        for i in range(args.power_iter):
            v = HTilde @ v
            v /= v.norm()
        lambda_max = v.T @ HTilde @ v
        print("lambda_max:", lambda_max, flush=True)
        lambda_max = lambda_max.view(-1)
        D = (lambda_max * DTilde).to(device)

        Wr = Y.to(device)
        HRr = HR.to(device)
        
        Wscale = Wr.square().mean().sqrt() / (
            cb.lut.to(torch.float64).square().mean().sqrt().float() *
            args.scale_override)
        Wr /= Wscale

        LRr, _ = utils.block_LDL(HRr, args.td_y)
        zeroLRr = torch.zeros_like(LRr, device=LRr.device)
        diag = torch.arange(n, device=LRr.device)
        LRr[diag, diag] = 0
        args.td_x = m
        args.td_y = n

        scalar = torch.zeros(m, n, device=device)

        for i in range(scalar.shape[0]):
            for j in range(scalar.shape[1]):
                scalar[i][j] = D1[i] * D1[i] * D2[j] * D2[j] * D[j]

        # apply random permutation to the flattened Wr and Scalar

        permutation = torch.randperm(m * n)
        
        Wr = Wr.flatten()[permutation].reshape(m, n)
        scalar = scalar.flatten()[permutation].reshape(m, n)
        
        hatWr, Qidxs = ldlq.LDLQ(Wr, zeroLRr, cb, args, for_kernel=has_kernel)


        Qidxs = Qidxs.cpu()
        packed = cb.pack_trellis(
            Qidxs.reshape(m // args.td_x, args.td_x, n // args.td_y,
                          args.td_y // args.V).transpose(1, 2).reshape(
                              -1, args.td_x * args.td_y // args.V))

        packed = packed.view(torch.int16)

        if rcp == 'col':
            Wr = (Wr.reshape(args.tp_rank, m * n // args.tp_rank) *
                  Wscale.unsqueeze(-1)).reshape(m, n)
            hatWr = (hatWr.reshape(args.tp_rank, m * n // args.tp_rank) *
                     Wscale.unsqueeze(-1)).reshape(m, n)
        elif rcp == 'row':
            Wr = Wr.reshape(m, args.tp_rank, n // args.tp_rank).transpose(
                0, 1).reshape(args.tp_rank, -1) * Wscale.unsqueeze(-1)
            Wr = Wr.reshape(args.tp_rank, m,
                            n // args.tp_rank).transpose(0, 1).reshape(m, n)
            hatWr = hatWr.reshape(m, args.tp_rank,
                                  n // args.tp_rank).transpose(0, 1).reshape(
                                      args.tp_rank, -1) * Wscale.unsqueeze(-1)
            hatWr = hatWr.reshape(args.tp_rank, m,
                                  n // args.tp_rank).transpose(0, 1).reshape(
                                      m, n)
        else:
            Wr *= Wscale
            hatWr *= Wscale
        
        inverse_permutation = torch.argsort(permutation)
        Wr = Wr.flatten()[inverse_permutation].reshape(m, n)
        hatWr = hatWr.flatten()[inverse_permutation].reshape(m, n)
        scalar = scalar.flatten()[inverse_permutation].reshape(m, n)
        print(f"Wr after initialize and scaling:", Wr, flush=True)
        print(f"hatWr after initialize and scaling:", hatWr, flush=True)
        squeezed_D1col = D1[:, None].to(device)
        squeezed_D1row = D1[None, :].to(device)
        squeezed_D2row = D2[None, :].to(device)
        squeezed_D2col = D2[:, None].to(device)
        err = torch.trace(
            (squeezed_D1col * (Wr - hatWr) * squeezed_D2row) @ HRr @ (squeezed_D2col * (Wr - hatWr).T * squeezed_D1row)) / torch.trace((squeezed_D1col * Wr * squeezed_D2row) @ HRr @ (squeezed_D2col * Wr.T * squeezed_D1row))
        print(
            f'initialize W_0 {idx}_{name} with LDLQ, proxy err {err.item()} tr(WHW.T) {torch.trace((squeezed_D1col * Wr * squeezed_D2row) @ HRr @ (squeezed_D2col * Wr.T * squeezed_D1row))}'
        )
        # update the weights by descending of proxy loss


        for i in range(args.update_iter):
            
            Wr /= Wscale
            hatWr /= Wscale


            V = (hatWr - ((hatWr - Wr) @ HRr) * (1 / D)[None, :])

            Wr = Wr.flatten()[permutation].reshape(m, n)
            hatWr = hatWr.flatten()[permutation].reshape(m, n)
            scalar = scalar.flatten()[permutation].reshape(m, n)
            V = V.flatten()[permutation].reshape(m, n)
            print(f"Wr before {i+1}-th update:", Wr, flush=True)
            print(f"hatWr before {i+1}-th update:", hatWr, flush=True)

            hatWr, Qidxs = ldlq.LDLQ(V, zeroLRr, cb, args, for_kernel=has_kernel, scalar=scalar)
            print(f"Wr after {i+1}-th update:", Wr, flush=True)
            print(f"hatWr after {i+1}-th update:", hatWr, flush=True)
            Qidxs = Qidxs.cpu()
            packed = cb.pack_trellis(
                Qidxs.reshape(m // args.td_x, args.td_x, n // args.td_y,
                            args.td_y // args.V).transpose(1, 2).reshape(
                                -1, args.td_x * args.td_y // args.V))


            packed = packed.view(torch.int16)

            if rcp == 'col':
                Wr = (Wr.reshape(args.tp_rank, m * n // args.tp_rank) *
                    Wscale.unsqueeze(-1)).reshape(m, n)
                hatWr = (hatWr.reshape(args.tp_rank, m * n // args.tp_rank) *
                        Wscale.unsqueeze(-1)).reshape(m, n)
            elif rcp == 'row':
                Wr = Wr.reshape(m, args.tp_rank, n // args.tp_rank).transpose(
                    0, 1).reshape(args.tp_rank, -1) * Wscale.unsqueeze(-1)
                Wr = Wr.reshape(args.tp_rank, m,
                                n // args.tp_rank).transpose(0, 1).reshape(m, n)
                hatWr = hatWr.reshape(m, args.tp_rank,
                                    n // args.tp_rank).transpose(0, 1).reshape(
                                        args.tp_rank, -1) * Wscale.unsqueeze(-1)
                hatWr = hatWr.reshape(args.tp_rank, m,
                                    n // args.tp_rank).transpose(0, 1).reshape(
                                        m, n)
            else:
                Wr *= Wscale
                hatWr *= Wscale

            Wr = Wr.flatten()[inverse_permutation].reshape(m, n)
            hatWr = hatWr.flatten()[inverse_permutation].reshape(m, n)
            scalar = scalar.flatten()[inverse_permutation].reshape(m, n)

            print(f"Wr after {i+1}-th update and scaling:", Wr, flush=True)
            print(f"hatWr after {i+1}-th update and scaling:", hatWr, flush=True)
            err = torch.trace(
            (squeezed_D1col * (Wr - hatWr) * squeezed_D2row) @ HRr @ (squeezed_D2col * (Wr - hatWr).T * squeezed_D1row)) / torch.trace((squeezed_D1col * Wr * squeezed_D2row) @ HRr @ (squeezed_D2col * Wr.T * squeezed_D1row))
            print(
                f'Updated W_{i+1} {idx}_{name} with LDLQ, proxy err {err.item()} tr(WHW.T) {torch.trace((squeezed_D1col * Wr * squeezed_D2row) @ HRr @ (squeezed_D2col * Wr.T * squeezed_D1row))}'
            )

        save_path = f'{args.save_path}/{idx}_{name}.pt'

        # 0 = no tensor parallelism, 1 = row parallel, 2 = column parallel
        rcp_int = 0
        if args.split_for_tp:
            rcp_int = 1 if rcp == 'row' else 2

        torch.save(
            {
                'trellis':
                packed.cpu(),
                'Wscale':
                Wscale,
                'proxy_err':
                err.item(),
                'tlut':
                cb.tlut.data.to(orig_dtype).cpu()
                if hasattr(cb, 'tlut') else None,
                'rcp':
                rcp_int,
                'tp_rank':
                args.tp_rank
            }, save_path)


        del HRr, Wr, hatWr, LRr, Qidxs
        utils.clean()
        
        q_linear = QuantizedLinear(
            n,
            m,
            args.td_x,
            args.td_y,
            args.L,
            args.K,
            args.V,
            args.tlut_bits,
            args.decode_mode,
            mode='train-recons' if args.ft_train_lut else 'train-fixW',
            dtype=orig_dtype,
            grad_ckpt=args.ft_grad_ckpt)
        q_linear.trellis.copy_(packed)
        # q_linear.SU.copy_(SU)
        # q_linear.SV.copy_(SV)
        q_linear.rcp.copy_(rcp_int)
        q_linear.tp_rank.copy_(args.tp_rank)
        q_linear = q_linear.to(device).float()

        # del packed, SU, SV
        utils.clean()
        
        if rcp == 'row':
            q_linear.SU = nn.Parameter(
                (q_linear.SU.reshape(args.tp_rank, -1) *
                 Wscale.unsqueeze(-1)).reshape(q_linear.SU.shape),
                requires_grad=True)
            q_linear.SV = nn.Parameter(q_linear.SV, requires_grad=True)
        elif rcp == 'col':
            q_linear.SU = nn.Parameter(q_linear.SU, requires_grad=True)
            q_linear.SV = nn.Parameter(
                (q_linear.SV.reshape(args.tp_rank, -1) *
                 Wscale.unsqueeze(-1)).reshape(q_linear.SV.shape),
                requires_grad=True)
        else:
            q_linear.SU = nn.Parameter(q_linear.SU, requires_grad=True)
            q_linear.SV = nn.Parameter(q_linear.SV * Wscale,
                                       requires_grad=True)

        if q_linear.tlut is not None:
            q_linear.tlut.copy_(cb.tlut.data)
            q_linear.tlut.requires_grad = args.ft_train_lut

        split_attr = linear_attr.split('.')
        setattr(
            attrgetter('.'.join(split_attr[:-1]))(mixed_layer), split_attr[-1],
            q_linear)

        # for tensor-wise fine-tuning
        # with torch.enable_grad():
        #     finetune_decoder_layer(mixed_layer, f'{idx}_{name}', device,
        #                            train_dl, valid_dl, orig_dtype, args)

        cb = cb.cpu()
        utils.clean()

    for quant_i, (linear_attr, name, in_hess_name, out_hess_name,
                  rcp) in enumerate(quant_order):
        quant_linear = attrgetter(linear_attr)(mixed_layer)
        save_path = f'{args.save_path}/{idx}_{name}.pt'
        data = torch.load(save_path)
        if rcp == 'row':
            data['SU'] = (
                ((quant_linear.SU.data).reshape(args.tp_rank, -1) /
                 data['Wscale'].to(quant_linear.SU.device).unsqueeze(-1)
                 ).reshape(quant_linear.SU.data.shape)).to(orig_dtype).cpu()
            data['SV'] = quant_linear.SV.data.to(orig_dtype).cpu()
        elif rcp == 'col':
            data['SU'] = quant_linear.SU.data.to(orig_dtype).cpu()
            data['SV'] = (
                ((quant_linear.SV.data).reshape(args.tp_rank, -1) /
                 data['Wscale'].to(quant_linear.SV.device).unsqueeze(-1)
                 ).reshape(quant_linear.SV.data.shape)).to(orig_dtype).cpu()
        else:
            data['SU'] = quant_linear.SU.data.to(orig_dtype).cpu()
            data['SV'] = (quant_linear.SV.data / data['Wscale'].to(
                quant_linear.SV.device)).to(orig_dtype).cpu()

        if quant_linear.tlut is not None:
            data['tlut'] = quant_linear.tlut.data.to(orig_dtype).cpu()
        torch.save(data, save_path)

    mixed_layer = mixed_layer.to(orig_dtype).cpu()

    utils.clean()
    torch.set_grad_enabled(False)


def infer(args, end_dev, n_layers, in_q, out_q):
    with torch.no_grad():
        fake_dev_map = {
            'model.embed_tokens': 0,
            'model.rotary_emb': 0,
            'model.norm': end_dev - 1,
            'lm_head': end_dev - 1
        }
        per_dev = math.ceil(n_layers / end_dev)
        for i in range(n_layers):
            fake_dev_map[f'model.layers.{i}'] = (i + 1) // per_dev

        model = AutoModelForCausalLM.from_pretrained(args.base_model,
                                                     torch_dtype='auto',
                                                     device_map=fake_dev_map,
                                                     low_cpu_mem_usage=True)
        while True:
            data = in_q.get()
            if data is None:
                return
            out_q.put(
                model(data.to(0))['logits'][:, :-1].contiguous().softmax(
                    dim=-1).cpu())


def finetune_susv_e2e(quant_model, start_dev, devset, orig_dtype, args):

    in_q = mp.Queue()
    out_q = mp.Queue()
    p = mp.Process(target=infer,
                   args=(args, start_dev, len(quant_model.model.layers), in_q,
                         out_q))
    p.start()

    train_dl, valid_dl = utils.split_data(devset, devset, args)

    optim = torch.optim.Adam(quant_model.parameters(), lr=args.ft_lr)

    best_loss = utils.calculate_ce_loss_model(quant_model, valid_dl, start_dev,
                                              in_q, out_q)
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    best_sd = copy.deepcopy(quant_model.state_dict())
    glog.info(f'initial loss {best_loss}')
    worse_ct = 0
    for epoch in range(args.ft_epochs):
        for bidx, (source, _) in enumerate(train_dl):
            in_q.put(source)
            with torch.autocast(device_type='cuda',
                                dtype=orig_dtype,
                                enabled=True):
                output = quant_model(
                    source.to(start_dev))['logits'][:, :-1].contiguous()
                target = out_q.get().to(output.device)
                target = target.view(-1, target.shape[-1])
                loss = nn.CrossEntropyLoss()(output.view(-1, output.shape[-1]),
                                             target)
            scaler.scale(loss).backward()
            if bidx % args.ft_update_freq == args.ft_update_freq - 1 or bidx == len(
                    train_dl) - 1:
                scaler.step(optim)
                scaler.update()
                optim.zero_grad()

        if epoch % args.ft_valid_freq == (args.ft_valid_freq - 1):
            test_loss = utils.calculate_ce_loss_model(quant_model, valid_dl,
                                                      start_dev, in_q, out_q)
            if test_loss < best_loss:
                glog.info(
                    f'epoch {epoch} new loss {test_loss} old loss {best_loss} BETTER'
                )
                best_loss = test_loss
                best_sd = copy.deepcopy(quant_model.state_dict())
                worse_ct = 0
            else:
                glog.info(
                    f'epoch {epoch} new loss {test_loss} old loss {best_loss} WORSE'
                )
                worse_ct += 1
                if worse_ct >= args.ft_early_stop:
                    break

    in_q.put(None)
    p.join()
    with torch.no_grad():
        quant_model.load_state_dict(best_sd)
