"""LIF with ROW-normalized weights (total input=1, E/I ratio preserved) + Gaussian background drive."""
import numpy as np, scipy.sparse as sp, time

d = np.load("graph.npz", allow_pickle=True)
N = int(d['N']); signs = d['signs']
W = sp.csr_matrix((d['data'], d['indices'], d['indptr']), shape=(N, N))
W.eliminate_zeros()

# row normalize: each neuron's total |input| weight = 1
row_sum = np.bincount(np.repeat(np.arange(N), np.diff(W.indptr)),
                      weights=np.abs(W.data), minlength=N).astype(np.float64)
row_sum[row_sum == 0] = 1.0
Wr = sp.diags(1.0/row_sum) @ W
Wr = Wr.tocsr()
print(f"N={N:,} nnz={Wr.nnz:,}")

dt = 0.5; tau = 20.0; Vth = 1.0; ref_steps = 4

def lif(g=2.0, mu=0.8, sig=0.12, T=4000, seed=0):
    rng = np.random.default_rng(seed)
    V = np.zeros(N)
    ref = np.zeros(N, dtype=np.int32)
    pop = np.zeros(T); spike_count = np.zeros(N)
    for t in range(T):
        spk = V >= Vth
        I = g * Wr.dot(spk.astype(np.float64))
        mask = ref == 0
        V[mask] += (dt/tau) * (-V[mask] + mu) + I[mask]
        V[mask] += sig * rng.standard_normal(int(mask.sum()))
        V[spk] = 0.0
        ref[spk] = ref_steps
        ref = np.maximum(ref - 1, 0)
        spike_count += spk; pop[t] = spk.mean()
        if (t+1) % 1000 == 0:
            fr = pop[max(0,t-300):t+1].mean()*1000
            print(f"  g={g} mu={mu}: t={t+1}, pop={fr:.1f} Hz, active={int((spike_count>0).sum())}", flush=True)
    return pop, spike_count

results = []
for g, mu, sig in [(2.0,0.8,0.12),(3.0,0.8,0.12),(4.0,0.8,0.12),(3.0,0.9,0.15),(3.0,0.85,0.10)]:
    print(f"\n===== g={g}, mu={mu}, sig={sig} =====")
    t0=time.time()
    pop, sc = lif(g, mu, sig)
    late = pop[1000:]
    firing = sc / (4000*dt/1000.0)
    fa = firing[firing>0]
    cv = late.std()/max(late.mean(),1e-12)
    print(f"  群体率 后期 {late.mean()*1000:.2f} Hz, Fano因子(变异系数^2) ≈ {cv:.2f}")
    print(f"  活跃神经元 {len(fa):,} ({len(fa)/N:.1%}), 平均 {fa.mean():.1f} Hz | 耗时 {time.time()-t0:.0f}s")
    results.append((g,mu,sig,late.mean()*1000, len(fa), cv))
    np.save(f"pop_lif_g{g}_mu{mu}.npy", pop)
    np.save(f"spikes_lif_g{g}_mu{mu}.npy", sc)
