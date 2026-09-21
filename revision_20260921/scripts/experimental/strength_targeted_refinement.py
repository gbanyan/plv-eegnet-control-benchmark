"""T-only candidate: bounded targeted edge swaps, preserving the exact multiset.

This is an explicit extension of the archived v2 sampler, not a retry hidden
inside that frozen protocol. The same stopping rule applies to every member.
"""
import hashlib
import json
import numpy as np


def named_seed(key):
    return int.from_bytes(hashlib.sha256(json.dumps(key,separators=(',',':')).encode()).digest()[:16],'little')


def strengths(w,u,v,n):
    return np.bincount(u,weights=w,minlength=n)+np.bincount(v,weights=w,minlength=n)


def refine(original,initial,key,*,maximum_steps=2048,maximum_relative=.025,rms_relative=.005):
    if maximum_steps<1 or maximum_relative<=0 or rms_relative<=0:
        raise ValueError("positive limits required")
    original=np.asarray(original,dtype=np.float64)
    w=np.asarray(initial,dtype=np.float64).copy()
    n=int((1+np.sqrt(1+8*len(w)))/2);u,v=np.triu_indices(n,1)
    if len(u)!=len(w) or not np.array_equal(np.sort(w),np.sort(original)):
        raise ValueError('triangular exact-multiset input required')
    target=strengths(original,u,v,n);scale=np.maximum(target,1.e-12)
    inverse=1/scale**2
    lookup=np.full((n,n),-1,dtype=int);lookup[u,v]=lookup[v,u]=np.arange(len(w))
    rng=np.random.Generator(np.random.PCG64(named_seed((*key,'targeted-relative-v3'))))
    neighbors=rng.integers(n-1,size=maximum_steps)
    accepted=0;initial_energy=float(np.sum(((strengths(w,u,v,n)-target)/scale)**2))
    for step,neighbor in enumerate(neighbors):
        # Recompute rather than accumulate node-strength roundoff.
        s=strengths(w,u,v,n);relative=(s-target)/scale
        if np.max(np.abs(relative))<=maximum_relative and np.sqrt(np.mean(relative**2))<=rms_relative:
            break
        a=int(np.argmax(np.abs(relative)));b=int(neighbor+(neighbor>=a));i=lookup[a,b]
        delta=w-w[i];gradient=(s-target)*inverse
        shared=((u==a)|(v==a))*inverse[a]+((u==b)|(v==b))*inverse[b]
        change=2*delta*(gradient[a]+gradient[b]-gradient[u]-gradient[v])+delta**2*(inverse[a]+inverse[b]+inverse[u]+inverse[v]-2*shared)
        j=int(np.argmin(change))
        if change[j]<-1.e-18:
            w[i],w[j]=w[j],w[i];accepted+=1
    else:step=maximum_steps
    relative=(strengths(w,u,v,n)-target)/scale
    energy=float(np.sum(relative**2))
    if energy>initial_energy+1.e-12:raise RuntimeError('nonmonotone independently recomputed energy')
    return w,dict(steps=int(step),accepted=accepted,initial_energy=initial_energy,final_energy=energy,
                  relative_max=float(np.max(np.abs(relative))),relative_rms=float(np.sqrt(np.mean(relative**2))))
