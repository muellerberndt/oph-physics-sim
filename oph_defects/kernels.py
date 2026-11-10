"""Native kernels for universe-scale law C runs: one attempt body, three execution orders.

The attempt body (``CORE``) is one piece of C that also compiles as CUDA. It
performs exactly the integer law of ``oph_exact.federation`` (law C: nearest
agreement with a fair tie coin) on one seam attempt and, on top of the
unchanged loads, the declared layers: causal depths, incident clocks,
consistent-cut copies and the energy-accounted Z6 link register.

Execution orders:

``sequential``   attempts in schedule order (the reference);
``layers``       the same attempts executed in causal layers on the CPU: an
                 attempt runs once no earlier pending attempt shares one of
                 its resources (its two ports, and for a glue seam its two
                 screen plaquettes). Within a layer attempts are disjoint and
                 commute, so every state array and every per-attempt output is
                 identical to the sequential order. The CPU version exists to
                 test the scheme; it can run each layer in reverse order;
``cuda``         the same layers on a GPU (CuPy RawKernel), disjoint attempts
                 of a layer in parallel.
"""
from __future__ import annotations

import ctypes
import hashlib
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np

CORE = r"""
#define OPH_INF 2147483647
#define OPH_F_REG 1
#define OPH_F_OFF 2
#define OPH_F_JSUM 4
#define OPH_F_COUNT 8
#define OPH_F_TWIST 16
typedef struct {
    int64_t N, P, SI, FI, F0;
    const int8_t *ta, *tb, *tp, *tm;          /* [30] template seam ends and face sides */
    const int32_t *ga, *gb, *gfp, *gfm;       /* [G] glue seam ends and plaquettes (global face ids) */
    int8_t *x;                                /* [P] loads */
    int8_t *reg;                              /* [S] Z6 link per seam */
    int8_t *cls;                              /* [F] class per face */
    int32_t *dnat;                            /* [P] native causal depth per port */
    int32_t *dsyn;                            /* [P + V] process depth per port, then per plaquette */
    int32_t *ticks;                           /* [N] incident attempts per carrier */
    const int32_t *cost;                      /* [6] */
    int32_t kappa;
    int32_t ncut;
    const int32_t *cutT;                      /* [ncut] ascending depths of the open cuts */
    const int32_t *cut_slot;                  /* [ncut] storage slot of each open cut */
    int8_t *cut_x;                            /* [slots * P] */
    int32_t *cut_ticks;                       /* [slots * N] */
    int8_t *o_cur, *o_con, *o_twist, *o_de;   /* [n] per attempt */
    int8_t *o_tot;                            /* [n] per attempt: x_a + x_b before the attempt */
    int32_t *o_dep;                           /* [n] per attempt */
    int32_t flags;                            /* OPH_F_* */
    int8_t *off;                              /* [S] seam offsets b_e, oriented a->b (OPH_F_OFF); changed by twists (OPH_F_TWIST) */
    int32_t *jsum;                            /* [S] cumulative current per seam (OPH_F_JSUM) */
    int32_t *nzc, *dsc;                       /* [N] nonzero-current and descent events per carrier (OPH_F_COUNT) */
    int32_t *events;                          /* [ev_cap * 8] */
    int64_t ev_cap;
    int64_t *ev_count;                        /* [2]: rows used, overflow flag */
} oph_ctx;

/* resources of seam s: its two ports, and for a glue seam its two plaquettes (indices into dsyn) */
OPH_FN int oph_resources(const oph_ctx *c, int64_t s, int64_t *r)
{
    if (s < c->SI) {
        const int64_t cc = s / 30; const int k = (int)(s - 30 * cc);
        r[0] = 12 * cc + c->ta[k]; r[1] = 12 * cc + c->tb[k];
        return 2;
    }
    const int64_t g = s - c->SI;
    r[0] = c->ga[g]; r[1] = c->gb[g];
    r[2] = c->P + ((int64_t)c->gfp[g] - c->FI); r[3] = c->P + ((int64_t)c->gfm[g] - c->FI);
    return 4;
}

/* one attempt: seam s with tie coin, written to output row i, logged with schedule position t */
OPH_FN void oph_attempt(oph_ctx *c, int64_t s, int coinbit, int64_t i, int64_t t)
{
    int64_t a, b, fp, fm, c1, c2 = -1, ra = 0, rb = 0;
    int glue = 0;
    if (s < c->SI) {
        const int64_t cc = s / 30; const int k = (int)(s - 30 * cc);
        a = 12 * cc + c->ta[k]; b = 12 * cc + c->tb[k];
        fp = 20 * cc + c->tp[k]; fm = 20 * cc + c->tm[k]; c1 = cc;
    } else {
        const int64_t g = s - c->SI;
        a = c->ga[g]; b = c->gb[g]; fp = c->gfp[g]; fm = c->gfm[g];
        c1 = a / 12; c2 = b / 12; glue = 1;
        ra = c->P + (fp - c->FI); rb = c->P + (fm - c->FI);
    }
    /* law C, exactly as oph_exact.federation: total, floor and ceiling, coin picks the first end.
       With OPH_F_OFF the seam compares x_a with the transported neighbour x_b + b_e (declared extension);
       b_e = 0 is law C. */
    const int xi = c->x[a], xj = c->x[b];
    const int bo = (c->flags & OPH_F_OFF) ? c->off[s] : 0;
    const int tot = xi + xj + bo, lo = (tot - (tot & 1)) / 2, hi = tot - lo;
    const int ni = coinbit ? hi : lo, nj = tot - ni - bo;
    const int d = xi - xj - bo, cur = xi - ni;
    /* causal depths: native (ports) and process (ports and plaquettes) */
    int32_t dn = c->dnat[a]; if (c->dnat[b] > dn) dn = c->dnat[b];
    dn += 1;
    int32_t ds = c->dsyn[a]; if (c->dsyn[b] > ds) ds = c->dsyn[b];
    if (glue) { if (c->dsyn[ra] > ds) ds = c->dsyn[ra]; if (c->dsyn[rb] > ds) ds = c->dsyn[rb]; }
    ds += 1;
    c->dnat[a] = dn; c->dnat[b] = dn; c->dsyn[a] = ds; c->dsyn[b] = ds;
    if (glue) { c->dsyn[ra] = ds; c->dsyn[rb] = ds; }
    /* incident clocks */
    OPH_INC32(&c->ticks[c1]);
    if (glue) OPH_INC32(&c->ticks[c2]);
    for (int q = c->ncut - 1; q >= 0 && c->cutT[q] >= ds; --q) {
        int32_t *ct = c->cut_ticks + (int64_t)c->cut_slot[q] * c->N;
        OPH_INC32(&ct[c1]);
        if (glue) OPH_INC32(&ct[c2]);
    }
    int twist = 0, de = 0, twisted = 0;
    if ((c->flags & OPH_F_TWIST) && glue && d != 0) {
        /* declared extension, phase 2: before moving load, the attempt asks whether correcting the seam offset by one
           unit pays for the change of the two plaquette holonomies. The twist releases the seam potential |d| - 1/2;
           cost is in units of kappa per unit of potential, extended additively beyond |q| = 3. */
        const int k = d > 0 ? 1 : -1;
        const int q1 = c->cls[fp], q2 = c->cls[fm];
        const int m1 = q1 + k, m2 = q2 - k;
#define OPH_CQ(q) (((q) < 0 ? -(q) : (q)) <= 3 ? c->cost[(q) < 0 ? -(q) : (q)] : c->cost[3] + (((q) < 0 ? -(q) : (q)) - 3) * c->cost[1])
        const int dc = OPH_CQ(m1) + OPH_CQ(m2) - OPH_CQ(q1) - OPH_CQ(q2);
#undef OPH_CQ
        const int lhs = 2 * dc, rhs = 2 * c->kappa * (d > 0 ? d : -d) - c->kappa;
        if (lhs < rhs || (lhs == rhs && coinbit)) {
            c->off[s] = (int8_t)(c->off[s] + k);
            c->cls[fp] = (int8_t)m1; c->cls[fm] = (int8_t)m2;
            twisted = 1; twist = k; de = dc;
            if (fp >= c->F0) {
                int64_t e;
                OPH_EVSLOT(e);
                if (e >= 0) {
                    int32_t *row = c->events + 8 * e;
                    row[0] = (int32_t)t; row[1] = (int32_t)fp; row[2] = (int32_t)fm; row[3] = q1; row[4] = q2;
                    row[5] = m1; row[6] = m2; row[7] = ds;
                }
            }
        }
    }
    if (!twisted && cur != 0) {
        c->x[a] = (int8_t)ni; c->x[b] = (int8_t)nj;
        for (int q = c->ncut - 1; q >= 0 && c->cutT[q] >= ds; --q) {
            int8_t *cx = c->cut_x + (int64_t)c->cut_slot[q] * c->P;
            cx[a] = (int8_t)ni; cx[b] = (int8_t)nj;
        }
        if (c->flags & OPH_F_JSUM) c->jsum[s] += cur;
        if (c->flags & OPH_F_COUNT) {
            OPH_INC32(&c->nzc[c1]); if (glue) OPH_INC32(&c->nzc[c2]);
            if (d >= 2 || d <= -2) { OPH_INC32(&c->dsc[c1]); if (glue) OPH_INC32(&c->dsc[c2]); }
        }
      if (c->flags & OPH_F_REG) {
        /* declared register: the moved current proposes a twist of the seam's Z6 link; it is applied
           when the mismatch potential released by this repair covers the curvature cost */
        const int delta = ((cur % 6) + 6) % 6;
        const int k1 = c->cls[fp], k2 = c->cls[fm];
        const int n1 = (k1 + delta) % 6, n2 = (k2 - delta + 6) % 6;
        de = c->cost[n1] + c->cost[n2] - c->cost[k1] - c->cost[k2];
        const int work = c->kappa * cur * (d - cur);
        if (de <= work) {
            c->reg[s] = (int8_t)((c->reg[s] + delta) % 6);
            c->cls[fp] = (int8_t)n1; c->cls[fm] = (int8_t)n2;
            twist = delta;
            if (fp >= c->F0) {
                int64_t e;
                OPH_EVSLOT(e);
                if (e >= 0) {
                    int32_t *row = c->events + 8 * e;
                    row[0] = (int32_t)t; row[1] = (int32_t)fp; row[2] = (int32_t)fm; row[3] = k1; row[4] = k2;
                    row[5] = n1; row[6] = n2; row[7] = ds;
                }
            }
        } else twist = -delta;
      }
    }
    c->o_cur[i] = (int8_t)(twisted ? 0 : cur); c->o_con[i] = (int8_t)d; c->o_twist[i] = (int8_t)twist; c->o_de[i] = (int8_t)de;
    c->o_tot[i] = (int8_t)(xi + xj); c->o_dep[i] = ds;
}
"""

C_PRELUDE = r"""
#include <stdint.h>
#include <string.h>
#define OPH_FN static inline
#define OPH_INC32(p) (++*(p))
#define OPH_EVSLOT(dst) { if (c->ev_count[0] < c->ev_cap) dst = c->ev_count[0]++; else { dst = -1; c->ev_count[1] = 1; } }
"""

C_ENTRY = r"""
/* cache hints for an attempt a few steps ahead; no effect on any result */
#define OPH_PF 16
static inline void oph_prefetch(const oph_ctx *c, int64_t s)
{
    if (s < c->SI) {
        const int64_t cc = s / 30;
        __builtin_prefetch(c->x + 12 * cc, 1); __builtin_prefetch(c->x + 12 * cc + 11, 1);
        __builtin_prefetch(c->dnat + 12 * cc, 1); __builtin_prefetch(c->dnat + 12 * cc + 11, 1);
        __builtin_prefetch(c->dsyn + 12 * cc, 1); __builtin_prefetch(c->dsyn + 12 * cc + 11, 1);
        __builtin_prefetch(c->cls + 20 * cc, 1); __builtin_prefetch(c->cls + 20 * cc + 19, 1);
        __builtin_prefetch(c->reg + s, 1); __builtin_prefetch(c->ticks + cc, 1);
    } else {
        const int64_t g = s - c->SI;
        __builtin_prefetch(c->ga + g, 0); __builtin_prefetch(c->gb + g, 0);
        __builtin_prefetch(c->gfp + g, 0); __builtin_prefetch(c->gfm + g, 0);
    }
}

/* schedule order: the reference */
void oph_sequential(oph_ctx *c, const int32_t *seq, const uint8_t *coin, int64_t n, int64_t t_base)
{
    for (int64_t i = 0; i < n; ++i) {
        if (i + OPH_PF < n) oph_prefetch(c, seq[i + OPH_PF]);
        oph_attempt(c, seq[i], coin[i], i, t_base + i);
    }
}

/* causal layers on the CPU. pend [n] and nxt [P + V] are work arrays; nxt must hold OPH_INF on entry
   and holds OPH_INF on exit. Returns the number of layers. */
int64_t oph_layers(oph_ctx *c, const int32_t *seq, const uint8_t *coin, int64_t n, int64_t t_base,
                   int32_t *pend, int32_t *nxt, int reverse)
{
    int64_t m = n, rounds = 0, r[4];
    for (int64_t i = 0; i < n; ++i) pend[i] = (int32_t)i;
    while (m > 0) {
        for (int64_t j = 0; j < m; ++j) {
            const int32_t i = pend[j];
            const int k = oph_resources(c, seq[i], r);
            for (int u = 0; u < k; ++u) if (i < nxt[r[u]]) nxt[r[u]] = i;
        }
        for (int64_t jj = 0; jj < m; ++jj) {
            const int64_t j = reverse ? m - 1 - jj : jj;
            const int32_t i = pend[j];
            const int k = oph_resources(c, seq[i], r);
            int ready = 1;
            for (int u = 0; u < k; ++u) if (nxt[r[u]] != i) ready = 0;
            if (ready) { oph_attempt(c, seq[i], coin[i], i, t_base + i); pend[j] = ~i; }
        }
        int64_t w = 0;
        for (int64_t j = 0; j < m; ++j) {
            const int32_t v = pend[j];
            const int32_t i = v < 0 ? ~v : v;
            const int k = oph_resources(c, seq[i], r);
            for (int u = 0; u < k; ++u) nxt[r[u]] = OPH_INF;
            if (v >= 0) pend[w++] = v;
        }
        m = w; ++rounds;
    }
    return rounds;
}

/* defect identities from ordered register events. ev rows: pos, fp, fm, k1, k2, n1, n2, depth.
   Returns the number of rows processed; stops early when fewer than two free catalogue rows remain. */
int64_t oph_track(const int32_t *ev, int64_t n, int32_t sweep, int64_t F0, int32_t *face_id,
                  uint8_t *c_cls, int32_t *c_bs, int32_t *c_bd, int32_t *c_ds, int32_t *c_dd,
                  int32_t *c_bface, int32_t *c_lface, int32_t *c_hops, int32_t *c_partner,
                  int32_t *c_p1, int32_t *c_p2, uint8_t *c_cause, int32_t *c_c1, int32_t *c_c2,
                  int64_t *n_ids, int64_t cap,
                  int32_t ncut, const int32_t *cutT, const int32_t *cut_slot, int8_t *cut_cls, int32_t *cut_fid,
                  int64_t FT, int64_t *stats)
{
    int64_t n_id = *n_ids;
#define NEWID(dst, klass, face, P1, P2) { dst = (int32_t)n_id++; c_cls[dst] = (uint8_t)(klass); c_bs[dst] = sweep; c_bd[dst] = dep; \
        c_ds[dst] = -1; c_dd[dst] = -1; c_bface[dst] = (int32_t)(face); c_lface[dst] = (int32_t)(face); c_hops[dst] = 0; \
        c_partner[dst] = -1; c_p1[dst] = (P1); c_p2[dst] = (P2); c_cause[dst] = 0; c_c1[dst] = -1; c_c2[dst] = -1; \
        face_id[(face) - F0] = dst; }
#define DIE(id, why, other) { c_ds[id] = sweep; c_dd[id] = dep; c_cause[id] = (uint8_t)(why); c_partner[id] = (other); }
    for (int64_t e = 0; e < n; ++e) {
        if (n_id + 2 > cap) { *n_ids = n_id; return e; }
        const int32_t *row = ev + 8 * e;
        const int64_t fp = row[1], fm = row[2];
        const int k1 = row[3], k2 = row[4], n1 = row[5], n2 = row[6];
        const int32_t dep = row[7];
        int32_t i1, i2, h, w;
        if (!k1 && !k2) {
            NEWID(h, n1, fp, -1, -1); NEWID(w, n2, fm, -1, -1);
            c_partner[h] = w; c_partner[w] = h; ++stats[0];
        } else if (k1 && !k2 && !n1) {
            i1 = face_id[fp - F0]; face_id[fm - F0] = i1; face_id[fp - F0] = -1; ++c_hops[i1]; c_lface[i1] = (int32_t)fm; ++stats[1];
        } else if (!k1 && k2 && !n2) {
            i2 = face_id[fm - F0]; face_id[fp - F0] = i2; face_id[fm - F0] = -1; ++c_hops[i2]; c_lface[i2] = (int32_t)fp; ++stats[1];
        } else if (k1 && !k2) {
            i1 = face_id[fp - F0]; DIE(i1, 3, -1);
            NEWID(h, n1, fp, i1, -1); NEWID(w, n2, fm, i1, -1);
            c_partner[h] = w; c_partner[w] = h; c_c1[i1] = h; c_c2[i1] = w; ++stats[4];
        } else if (!k1 && k2) {
            i2 = face_id[fm - F0]; DIE(i2, 3, -1);
            NEWID(h, n2, fm, i2, -1); NEWID(w, n1, fp, i2, -1);
            c_partner[h] = w; c_partner[w] = h; c_c1[i2] = h; c_c2[i2] = w; ++stats[4];
        } else {
            i1 = face_id[fp - F0]; i2 = face_id[fm - F0];
            if (!n1 && !n2) {
                DIE(i1, 1, i2); DIE(i2, 1, i1); face_id[fp - F0] = -1; face_id[fm - F0] = -1; ++stats[2];
            } else if (!n1) {
                DIE(i1, 2, i2); DIE(i2, 2, i1); face_id[fp - F0] = -1;
                NEWID(h, n2, fm, i1, i2); c_c1[i1] = h; c_c1[i2] = h; ++stats[3];
            } else if (!n2) {
                DIE(i1, 2, i2); DIE(i2, 2, i1); face_id[fm - F0] = -1;
                NEWID(h, n1, fp, i1, i2); c_c1[i1] = h; c_c1[i2] = h; ++stats[3];
            } else {
                DIE(i1, 4, i2); DIE(i2, 4, i1);
                NEWID(h, n1, fp, i1, i2); NEWID(w, n2, fm, i2, i1);
                c_c1[i1] = h; c_c2[i1] = w; c_c1[i2] = h; c_c2[i2] = w; ++stats[5];
            }
        }
        for (int q = ncut - 1; q >= 0 && cutT[q] >= dep; --q) {
            int8_t *cc = cut_cls + (int64_t)cut_slot[q] * FT;
            int32_t *cf = cut_fid + (int64_t)cut_slot[q] * FT;
            cc[fp - F0] = (int8_t)n1; cc[fm - F0] = (int8_t)n2;
            cf[fp - F0] = face_id[fp - F0]; cf[fm - F0] = face_id[fm - F0];
        }
    }
#undef NEWID
#undef DIE
    *n_ids = n_id;
    return n;
}
"""


C_RNG = r"""
/* numpy.random.Generator(PCG64).integers(0, bound, dtype=int64) for bound <= 2^32 - 1, reproduced exactly:
   PCG64 XSL-RR 128/64, 32-bit halves served low half first, Lemire bounded rejection as in numpy's
   buffered_bounded_lemire_uint32. st = {state_hi, state_lo, inc_hi, inc_lo}; buf = {has_uint32, uinteger}. */
typedef unsigned __int128 oph_u128;
static inline uint64_t oph_pcg_next64(oph_u128 *state, oph_u128 inc)
{
    const oph_u128 mult = ((oph_u128)2549297995355413924ULL << 64) | (oph_u128)4865540595714422341ULL;
    *state = *state * mult + inc;
    const uint64_t hi = (uint64_t)(*state >> 64), lo = (uint64_t)*state;
    const uint64_t v = hi ^ lo;
    const unsigned rot = (unsigned)(*state >> 122);
    return (v >> rot) | (v << ((-rot) & 63));
}
#define OPH_NEXT32(dst) { if (has) { has = 0; dst = held; } else { const uint64_t w_ = oph_pcg_next64(&state, inc); \
    has = 1; held = (uint32_t)(w_ >> 32); dst = (uint32_t)w_; } }
void oph_pcg_bounded(uint64_t *st, uint32_t *buf, uint32_t rng, int64_t n, int32_t *out32, uint8_t *out8)
{
    oph_u128 state = ((oph_u128)st[0] << 64) | st[1];
    const oph_u128 inc = ((oph_u128)st[2] << 64) | st[3];
    uint32_t has = buf[0], held = buf[1];
    const uint32_t rng_excl = rng + 1U;
    for (int64_t i = 0; i < n; ++i) {
        uint32_t u; OPH_NEXT32(u);
        uint64_t m = (uint64_t)u * rng_excl;
        uint32_t leftover = (uint32_t)m;
        if (leftover < rng_excl) {
            const uint32_t threshold = (0xFFFFFFFFU - rng) % rng_excl;
            while (leftover < threshold) { OPH_NEXT32(u); m = (uint64_t)u * rng_excl; leftover = (uint32_t)m; }
        }
        if (out32) out32[i] = (int32_t)(m >> 32); else out8[i] = (uint8_t)(m >> 32);
    }
    st[0] = (uint64_t)(state >> 64); st[1] = (uint64_t)state; buf[0] = has; buf[1] = held;
}
"""

CUDA_PRELUDE = r"""
typedef signed char int8_t;
typedef unsigned char uint8_t;
typedef int int32_t;
typedef long long int64_t;
#define OPH_FN __device__ __forceinline__
#define OPH_INC32(p) atomicAdd((int *)(p), 1)
#define OPH_EVSLOT(dst) { long long q_ = (long long)atomicAdd((unsigned long long *)&c->ev_count[0], 1ULL); \
    if (q_ < c->ev_cap) dst = q_; else { dst = -1; c->ev_count[1] = 1; } }
"""

CUDA_ENTRY = r"""
#define OPH_CTX_ARGS \
    long long N, long long P, long long SI, long long FI, long long F0, \
    const int8_t *ta, const int8_t *tb, const int8_t *tp, const int8_t *tm, \
    const int32_t *ga, const int32_t *gb, const int32_t *gfp, const int32_t *gfm, \
    int8_t *x, int8_t *reg, int8_t *cls, int32_t *dnat, int32_t *dsyn, int32_t *ticks, \
    const int32_t *cost, int kappa, int ncut, const int32_t *cutT, const int32_t *cut_slot, \
    int8_t *cut_x, int32_t *cut_ticks, int8_t *o_cur, int8_t *o_con, int8_t *o_twist, int8_t *o_de, int32_t *o_dep, \
    int32_t *events, long long ev_cap, int64_t *ev_count, \
    int8_t *o_tot, int flags, int8_t *off, int32_t *jsum, int32_t *nzc, int32_t *dsc
#define OPH_CTX_FILL oph_ctx cs; oph_ctx *c = &cs; \
    cs.N = N; cs.P = P; cs.SI = SI; cs.FI = FI; cs.F0 = F0; cs.ta = ta; cs.tb = tb; cs.tp = tp; cs.tm = tm; \
    cs.ga = ga; cs.gb = gb; cs.gfp = gfp; cs.gfm = gfm; cs.x = x; cs.reg = reg; cs.cls = cls; cs.dnat = dnat; \
    cs.dsyn = dsyn; cs.ticks = ticks; cs.cost = cost; cs.kappa = kappa; cs.ncut = ncut; cs.cutT = cutT; \
    cs.cut_slot = cut_slot; cs.cut_x = cut_x; cs.cut_ticks = cut_ticks; cs.o_cur = o_cur; cs.o_con = o_con; \
    cs.o_twist = o_twist; cs.o_de = o_de; cs.o_dep = o_dep; cs.events = events; cs.ev_cap = ev_cap; cs.ev_count = ev_count; \
    cs.o_tot = o_tot; cs.flags = flags; cs.off = off; cs.jsum = jsum; cs.nzc = nzc; cs.dsc = dsc;

extern "C" __global__ void oph_mark(const int32_t *pend, long long m, const int32_t *seq, int32_t *nxt, OPH_CTX_ARGS)
{
    const long long j = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= m) return;
    OPH_CTX_FILL
    const int32_t i = pend[j];
    int64_t r[4];
    const int k = oph_resources(c, seq[i], r);
    for (int u = 0; u < k; ++u) atomicMin((int *)&nxt[r[u]], (int)i);
}

extern "C" __global__ void oph_exec(const int32_t *pend, long long m, const int32_t *seq, const uint8_t *coin,
                                    long long t_base, const int32_t *nxt, uint8_t *done, OPH_CTX_ARGS)
{
    const long long j = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= m) return;
    OPH_CTX_FILL
    const int32_t i = pend[j];
    int64_t r[4];
    const int k = oph_resources(c, seq[i], r);
    int ready = 1;
    for (int u = 0; u < k; ++u) if (nxt[r[u]] != i) ready = 0;
    done[j] = (uint8_t)ready;
    if (ready) oph_attempt(c, seq[i], coin[i], i, t_base + i);
}

extern "C" __global__ void oph_reset(const int32_t *pend, long long m, const int32_t *seq, int32_t *nxt, OPH_CTX_ARGS)
{
    const long long j = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= m) return;
    OPH_CTX_FILL
    int64_t r[4];
    const int k = oph_resources(c, seq[pend[j]], r);
    for (int u = 0; u < k; ++u) nxt[r[u]] = OPH_INF;
}
"""

C_SOURCE = C_PRELUDE + CORE + C_ENTRY + C_RNG
CUDA_SOURCE = CUDA_PRELUDE + CORE + CUDA_ENTRY
CORE_SHA256 = hashlib.sha256(CORE.encode()).hexdigest()
C_SHA256 = hashlib.sha256(C_SOURCE.encode()).hexdigest()
INF = 2147483647
F_REG, F_OFF, F_JSUM, F_COUNT, F_TWIST = 1, 2, 4, 8, 16


class Ctx(ctypes.Structure):
    _fields_ = [('N', ctypes.c_int64), ('P', ctypes.c_int64), ('SI', ctypes.c_int64), ('FI', ctypes.c_int64),
                ('F0', ctypes.c_int64),
                ('ta', ctypes.c_void_p), ('tb', ctypes.c_void_p), ('tp', ctypes.c_void_p), ('tm', ctypes.c_void_p),
                ('ga', ctypes.c_void_p), ('gb', ctypes.c_void_p), ('gfp', ctypes.c_void_p), ('gfm', ctypes.c_void_p),
                ('x', ctypes.c_void_p), ('reg', ctypes.c_void_p), ('cls', ctypes.c_void_p), ('dnat', ctypes.c_void_p),
                ('dsyn', ctypes.c_void_p), ('ticks', ctypes.c_void_p), ('cost', ctypes.c_void_p),
                ('kappa', ctypes.c_int32), ('ncut', ctypes.c_int32), ('cutT', ctypes.c_void_p),
                ('cut_slot', ctypes.c_void_p), ('cut_x', ctypes.c_void_p), ('cut_ticks', ctypes.c_void_p),
                ('o_cur', ctypes.c_void_p), ('o_con', ctypes.c_void_p), ('o_twist', ctypes.c_void_p),
                ('o_de', ctypes.c_void_p), ('o_tot', ctypes.c_void_p), ('o_dep', ctypes.c_void_p),
                ('flags', ctypes.c_int32), ('off', ctypes.c_void_p), ('jsum', ctypes.c_void_p),
                ('nzc', ctypes.c_void_p), ('dsc', ctypes.c_void_p), ('events', ctypes.c_void_p),
                ('ev_cap', ctypes.c_int64), ('ev_count', ctypes.c_void_p)]


_LIB = {}


def library():
    """Compile (once per source hash) and load the C kernels."""
    if 'lib' in _LIB:
        return _LIB['lib']
    directory = Path(os.environ.get('OPH_EXACT_KERNEL_DIR', Path.home() / '.cache' / 'oph_exact_kernels'))
    directory.mkdir(parents=True, exist_ok=True)
    lib_path = directory / f'oph_universe_{C_SHA256[:16]}.so'
    if not lib_path.exists():
        compiler = shutil.which('cc') or shutil.which('clang') or shutil.which('gcc')
        if compiler is None:
            raise RuntimeError('no C compiler found')
        source = directory / f'oph_universe_{C_SHA256[:16]}.c'
        source.write_text(C_SOURCE)
        staging = directory / f'oph_universe_{C_SHA256[:16]}.{os.getpid()}.so'
        subprocess.run([compiler, '-O2', '-ffp-contract=off', '-shared', '-fPIC', '-o', str(staging), str(source)],
                       check=True, capture_output=True)
        os.replace(staging, lib_path)
    lib = ctypes.CDLL(str(lib_path))
    vp, i64 = ctypes.c_void_p, ctypes.c_int64
    lib.oph_sequential.argtypes = [ctypes.POINTER(Ctx), vp, vp, i64, i64]
    lib.oph_sequential.restype = None
    lib.oph_layers.argtypes = [ctypes.POINTER(Ctx), vp, vp, i64, i64, vp, vp, ctypes.c_int]
    lib.oph_layers.restype = i64
    lib.oph_track.argtypes = [vp, i64, ctypes.c_int32, i64, vp] + [vp] * 14 + [vp, i64, ctypes.c_int32, vp, vp, vp, vp, i64, vp]
    lib.oph_track.restype = i64
    lib.oph_pcg_bounded.argtypes = [vp, vp, ctypes.c_uint32, i64, vp, vp]
    lib.oph_pcg_bounded.restype = None
    _LIB['lib'] = lib
    return lib


def ptr(a: np.ndarray) -> int:
    return a.ctypes.data


class FastPCG64:
    """The draws of ``numpy.random.default_rng(seed).integers(0, bound, size, dtype=int64)`` from a C loop.

    The state lives in a numpy ``Generator`` (so checkpoints stay numpy states); each call copies the
    state out, fills the request in C and writes the advanced state back. ``verify`` compares against
    numpy itself and is called by the tests and at run time on a prefix of every sweep.
    """

    def __init__(self, generator: np.random.Generator):
        if generator.bit_generator.state['bit_generator'] != 'PCG64':
            raise ValueError('the schedule generator must be PCG64')
        self.gen = generator
        self.lib = library()

    def _load(self):
        st = self.gen.bit_generator.state
        s, inc = int(st['state']['state']), int(st['state']['inc'])
        mask = (1 << 64) - 1
        a = np.array([s >> 64, s & mask, inc >> 64, inc & mask], dtype=np.uint64)
        b = np.array([int(st['has_uint32']), int(st['uinteger'])], dtype=np.uint32)
        return a, b

    def _store(self, a, b):
        st = self.gen.bit_generator.state
        st['state']['state'] = (int(a[0]) << 64) | int(a[1])
        st['has_uint32'] = int(b[0])
        st['uinteger'] = int(b[1])
        self.gen.bit_generator.state = st

    def integers32(self, bound: int, n: int) -> np.ndarray:
        if not 2 <= bound <= 0x7FFFFFFF:
            raise ValueError('bound outside the 32-bit path')
        a, b = self._load()
        out = np.empty(n, np.int32)
        self.lib.oph_pcg_bounded(ptr(a), ptr(b), ctypes.c_uint32(bound - 1), n, ptr(out), None)
        self._store(a, b)
        return out

    def bits8(self, n: int) -> np.ndarray:
        a, b = self._load()
        out = np.empty(n, np.uint8)
        self.lib.oph_pcg_bounded(ptr(a), ptr(b), ctypes.c_uint32(1), n, None, ptr(out))
        self._store(a, b)
        return out


def verify_fast_rng(seed: int, bound: int, n: int) -> bool:
    """True when the C generator reproduces numpy's seam and coin draws and its end state."""
    ref = np.random.default_rng(seed)
    a = ref.integers(0, bound, size=n, dtype=np.int64)
    c = ref.integers(0, 2, size=n, dtype=np.int64)
    a2 = ref.integers(0, bound, size=7, dtype=np.int64)
    fast = FastPCG64(np.random.default_rng(seed))
    fa = fast.integers32(bound, n)
    fc = fast.bits8(n)
    fa2 = fast.integers32(bound, 7)
    return bool((fa == a).all() and (fc == c).all() and (fa2 == a2).all()
                and fast.gen.bit_generator.state == ref.bit_generator.state)
