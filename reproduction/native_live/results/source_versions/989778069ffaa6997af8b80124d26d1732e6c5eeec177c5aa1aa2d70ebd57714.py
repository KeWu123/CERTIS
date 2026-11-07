"""Vectorized, checked-int64 implementation of exact piecewise profiles.

Root neighbours use integer isqrt, never floating-point proposals. Unsafe
integer ranges fall back to the arbitrary-precision piecewise reference.
"""
from fractions import Fraction
from itertools import combinations
from math import isqrt

import numpy as np

from count_mask_planner import CountProfile, SideCounts, _fraction, _numpy_profile
from piecewise_count_profile import piecewise_profile


def _safe(left, right, q):
    # Each affine intercept is bounded by the represented count population,
    # slopes by one, and each polynomial is a difference of two products.
    # This bound also covers pair differences, discriminants and evaluation
    # at clipped feasible x. Compute the guard using Python integers.
    n=left.positive+left.unknown+right.positive+right.unknown
    return 32768*q*q*(n+1)*(n+1) <= np.iinfo(np.int64).max


def _affines(side, intercept, slope, midpoint2):
    low=max(side.lower,side.positive)
    high=min(side.upper,side.positive+side.unknown)
    value2=2*intercept+slope*midpoint2
    lost=(np.where(value2<=2*(high-side.positive),intercept,high-side.positive),
          np.where(value2<=2*(high-side.positive),slope,0))
    retained=tuple((np.where(value2<=2*(t-side.positive),t-intercept,side.positive),
                    np.where(value2<=2*(t-side.positive),-slope,0)) for t in (low,high))
    return lost,retained


def _mul(a,b):return a[0]*b[0],a[0]*b[1]+a[1]*b[0],a[1]*b[1]
def _add(a,b):return a[0]+b[0],a[1]+b[1]


def _root_arrays(polynomial):
    c,b,a=polynomial
    sign=np.where(a<0,-1,1);a,b,c=a*sign,b*sign,c*sign
    discriminant=b*b-4*a*c
    roots=np.zeros_like(discriminant)
    mask=(a!=0)&(discriminant>=0)
    roots[mask]=np.fromiter(map(isqrt,discriminant[mask]),dtype=np.int64,count=int(mask.sum()))
    denominator=np.where(a!=0,2*a,1)
    linear=(-c)//np.where(b!=0,b,1)
    plus=np.where(a!=0,(-b+roots)//denominator,linear)
    ceiling=roots+(roots*roots!=discriminant)
    minus=np.where(a!=0,(-b-ceiling)//denominator,linear)
    # For absent roots these arrays provide harmless additional candidates.
    # Every evaluated candidate is clipped to the feasible integer interval.
    return minus,plus


def numpy_piecewise_profile(left: SideCounts,right: SideCounts,allowance: Fraction,
                             *, small_left_cutoff=1024) -> CountProfile:
    """Return the same exact values and lexicographic witnesses as enumeration.

    The fixed small-left cutoff retains short vectorized enumeration loops.
    Its O(cutoff*n_R) work preserves the overall linear operation bound for
    a fixed cutoff; it does not change the mathematical profile or decisions.
    """
    if not isinstance(left,SideCounts) or not isinstance(right,SideCounts):raise TypeError('Two SideCounts required')
    p,q=_fraction(allowance)
    if not _safe(left,right,q):return piecewise_profile(left,right,allowance)
    if left.unknown<=small_left_cutoff:return _numpy_profile(left,right,p,q)
    forced=left.forced_discard+right.forced_discard
    g=np.arange(forced,left.unknown+right.unknown+1,dtype=np.int64)
    lower=np.maximum(left.forced_discard,g-right.unknown)
    upper=np.minimum(left.unknown,g-right.forced_discard)
    low_l=max(left.lower,left.positive)-left.positive
    high_l=min(left.upper,left.positive+left.unknown)-left.positive
    low_r=max(right.lower,right.positive)-right.positive
    high_r=min(right.upper,right.positive+right.unknown)-right.positive
    cuts=np.sort(np.stack((lower,upper,np.clip(np.full_like(g,low_l),lower,upper),
        np.clip(np.full_like(g,high_l),lower,upper),np.clip(g-low_r,lower,upper),
        np.clip(g-high_r,lower,upper)),axis=1),axis=1)
    best=np.full_like(g,np.iinfo(np.int64).max);chosen=lower.copy()
    for piece in range(5):
        lo,hi=cuts[:,piece],cuts[:,piece+1]
        a,bs=_affines(left,np.zeros_like(g),1,lo+hi)
        c,ds=_affines(right,g,-1,lo+hi)
        polynomials=[]
        for b in bs:
            for d in ds:
                total=_mul(_add(a,b),_add(c,d));retained=_mul(b,d)
                polynomials.append(tuple((q-p)*v-q*w for v,w in zip(total,retained)))

        def consider(candidate):
            x=np.clip(candidate,lo,hi)
            v=np.full_like(g,np.iinfo(np.int64).min)
            for c0,b0,a0 in polynomials:v=np.maximum(v,(a0*x+b0)*x+c0)
            better=(v<best)|((v==best)&(x<chosen))
            best[better]=v[better];chosen[better]=x[better]

        consider(lo);consider(hi)
        for c0,b0,a0 in polynomials:
            station=(-b0)//np.where(a0!=0,2*a0,1)
            consider(station);consider(station+1)
        for first,second in combinations(polynomials,2):
            for root in _root_arrays(tuple(v-w for v,w in zip(first,second))):
                consider(root);consider(root+1)
    # Independently recompute the four-pair support at every chosen witness.
    dl,dr=chosen,g-chosen
    lt0=np.minimum(min(left.upper,left.positive+left.unknown),np.maximum(max(left.lower,left.positive),dl+left.positive))
    lt1=np.full_like(g,min(left.upper,left.positive+left.unknown))
    rt0=np.minimum(min(right.upper,right.positive+right.unknown),np.maximum(max(right.lower,right.positive),dr+right.positive))
    rt1=np.full_like(g,min(right.upper,right.positive+right.unknown))
    actual=np.full_like(g,np.iinfo(np.int64).min)
    for lt in (lt0,lt1):
        for rt in (rt0,rt1):actual=np.maximum(actual,(q-p)*lt*rt-q*np.maximum(left.positive,lt-dl)*np.maximum(right.positive,rt-dr))
    if not np.array_equal(best,actual):raise AssertionError('Vectorized polynomial witness disagrees with four-pair kernel')
    return CountProfile(forced,tuple(map(int,best)),tuple(zip(map(int,dl),map(int,dr))))
