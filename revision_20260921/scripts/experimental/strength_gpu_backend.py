"""Versioned GPU annealing backend, independent of frozen CPU v1 sources."""
import time
import hashlib
import json
import numpy as np
import torch
import triton
import triton.language as tl
import triton.language.extra.cuda.libdevice as libdevice


def stable_seed(*key):
    """128-bit named PCG64 seed, independent of execution grouping."""
    return int.from_bytes(hashlib.sha256(json.dumps(key, separators=(',', ':')).encode()).digest()[:16], 'little')


@triton.jit
def kernel(W, BEST, TARGET, STRENGTH, ENERGY, MINIMUM, U, V, E1, E2, PROB,
           M: tl.constexpr, N: tl.constexpr, STEPS: tl.constexpr,
           TEMP, RELATIVE: tl.constexpr, B: tl.constexpr):
    chain = tl.program_id(0)
    nodes = tl.arange(0, N)
    edges = tl.arange(0, B)
    sb = tl.load(STRENGTH + chain*N + nodes)
    target = tl.load(TARGET + chain*N + nodes)
    energy = tl.load(ENERGY + chain)
    minimum = tl.load(MINIMUM + chain)
    temperature = tl.load(TEMP)
    for step in range(STEPS):
        k = chain*STEPS + step
        e1 = tl.load(E1+k); e2 = tl.load(E2+k)
        a = tl.load(U+e1); b = tl.load(V+e1)
        c = tl.load(U+e2); d = tl.load(V+e2)
        w1 = tl.load(W+chain*M+e1); w2 = tl.load(W+chain*M+e2)
        proposed = tl.where((nodes == a) | (nodes == b), (sb-w1)+w2, sb)
        proposed = tl.where((nodes == c) | (nodes == d), (proposed+w1)-w2, proposed)
        diff = proposed-target
        if RELATIVE:
            diff = diff / tl.maximum(target, 1.e-12)
        candidate = tl.sum(diff*diff, 0)
        prob = tl.load(PROB+k)
        accept = (candidate < energy) | (prob < libdevice.exp(-(candidate-energy)/temperature))
        if accept:
            sb = proposed
            energy = candidate
            tl.store(W+chain*M+e1, w2)
            tl.store(W+chain*M+e2, w1)
            tl.debug_barrier()
            if candidate < minimum:
                minimum = candidate
                weights = tl.load(W+chain*M+edges, edges < M, 0)
                tl.store(BEST+chain*M+edges, weights, edges < M)
                tl.debug_barrier()
    tl.store(STRENGTH+chain*N+nodes, sb)
    tl.store(ENERGY+chain, energy)
    tl.store(MINIMUM+chain, minimum)


def matrices(values):
    values=np.asarray(values,dtype=np.float64)
    n=int((1+np.sqrt(1+8*values.shape[-1]))/2)
    uv=np.triu_indices(n,1)
    a=np.zeros((len(values),n,n),dtype=np.float64)
    a[:,uv[0],uv[1]]=values
    return a+a.transpose(0,2,1)


def stage_tapes(keys, phase_name, stage, m, steps):
    # A named key identifies a surrogate independently of batch size/order.
    first=np.empty((len(keys),steps),dtype=np.int32)
    second=np.empty_like(first);probs=np.empty((len(keys),steps),dtype=np.float64)
    for i,key in enumerate(keys):
        rng=np.random.Generator(np.random.PCG64(stable_seed(*key,phase_name,stage)))
        first[i]=rng.integers(m,size=steps,dtype=np.int32)
        second[i]=rng.integers(m,size=steps,dtype=np.int32)
        probs[i]=rng.random(steps)
    return first,second,probs


def generate(originals, keys, phases):
    start=time.monotonic()
    originals=np.ascontiguousarray(originals,dtype=np.float64)
    if len(keys)!=len(originals) or len(set(keys))!=len(keys):
        raise ValueError('one unique key is required for each surrogate')
    current=np.stack([np.random.default_rng(stable_seed(*key,'initial')).permutation(w)
                      for key,w in zip(keys,originals)])
    target=matrices(originals).sum(2)
    n=target.shape[1];m=originals.shape[1];uv=np.triu_indices(n,1)
    gpu=lambda a:torch.from_numpy(np.ascontiguousarray(a)).cuda()
    target_gpu=gpu(target);u=gpu(uv[0].astype(np.int32));v=gpu(uv[1].astype(np.int32))
    reports=[]
    for phase in phases:
        began=time.monotonic()
        sb=matrices(current).sum(2)
        diff=sb-target
        if phase['relative']:diff=diff/np.maximum(target,1.e-12)
        energy=np.sum(diff*diff,axis=1)
        w=gpu(current);best=w.clone();s=gpu(sb);en=gpu(energy);minimum=en.clone()
        steps=phase['iterations']
        e1=torch.empty((len(keys),steps),dtype=torch.int32,device='cuda')
        e2=torch.empty_like(e1);prob=torch.empty((len(keys),steps),dtype=torch.float64,device='cuda')
        temperatures=np.array([phase['temperature']*phase['cooling']**stage
                               for stage in range(phase['stages'])],dtype=np.float64)
        if not np.isfinite(temperatures).all() or (temperatures<=0).any():
            raise ValueError('temperature schedule must remain positive in float64')
        temperatures_gpu=gpu(temperatures)
        for stage in range(phase['stages']):
            x,y,p=stage_tapes(keys,phase['name'],stage,m,steps)
            e1.copy_(torch.from_numpy(x));e2.copy_(torch.from_numpy(y));prob.copy_(torch.from_numpy(p))
            kernel[(len(keys),)](w,best,target_gpu,s,en,minimum,u,v,e1,e2,prob,
                                m,n,steps,temperatures_gpu[stage:stage+1],
                                phase['relative'],triton.next_power_of_2(m),
                                num_warps=4,enable_fp_fusion=False)
        torch.cuda.synchronize()
        current=best.cpu().numpy()
        reported=minimum.cpu().numpy()
        error=matrices(current).sum(2)-target
        if phase['relative']:error=error/np.maximum(target,1.e-12)
        actual=np.sum(error*error,axis=1)
        if not np.allclose(reported,actual,rtol=1.e-6,atol=1.e-8):
            raise RuntimeError('independently recomputed phase energy mismatch')
        reports.append(dict(name=phase['name'],seconds=time.monotonic()-began,
                            minimum_temperature=float(temperatures.min()),
                            reported_energy=reported.tolist(),actual_energy=actual.tolist()))
    return current,dict(seconds=time.monotonic()-start,phases=reports)
