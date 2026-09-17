"""LIF spiking dynamics on the column-normalized MaleCNS connectome. Look for self-sustained irregular firing."""
import numpy as np, scipy.sparse as sp, time

d = np.load("graph.npz", allow_pickle=True)
N = int(d['N']); signs = d['signs']
W = sp.csr_matrix((d['data'], d['indices'], d['indptr']), shape=(N, N))
W.eliminate_zeros()
# column normalize
col = np.bincount(W.indices, weights=np.abs(W.data), minlength=N).astype(np.float64)
s = np.zeros(N); s[col > 0] = 1.0/col[col > 0]
Wc = W @ sp.diags(s); Wc = Wc.tocsr()
print(f"N={N:,} nnz={Wc.nnz:,}")

dt = 0.5; tau = 20.0; Vth = 1.0; Vrest = 0.0; ref_steps = 4

def lif(g=1.5, nu_bg=4.0, T=4000, seed=0):
    rng = np.random.default_rng(seed)
    V = np.zeros(N)
    ref = np.zeros(N, dtype=np.int32)
    pop = np.zeros(T)          # 群体放电率
    spike_count = np.zeros(N)
    active_steps = np.zeros(T, dtype=bool)
    for t in range(T):
        spk = V >= Vth
        if spk.any():
            I = Wc.dot(spk.astype(np.float64)) * g
        else:
            I = 0.0
        # 背景泊松驱动
        bg = rng.random(N) < nu_bg * dt / 1000.0
        I = I + bg * 0.05
        # 膜电位更新(非不应期)
        mask = ref == 0
        V[mask] = V[mask] + dt/tau * (-V[mask]) + (I[mask] if np.isscalar(I) else I[mask])
        # 复位与不应期
        V[spk] = Vrest
        ref[spk] = ref_steps
        ref = np.maximum(ref - 1, 0)
        spike_count += spk
        pop[t] = spk.mean()
        active_steps[t] = spk.any()
        if (t+1) % 1000 == 0:
            print(f"  g={g} nu={nu_bg}: t={t+1}, pop_rate={pop[max(0,t-200):t+1].mean()*1000:.1f} Hz, "
                  f"active_neurons={int((spike_count>0).sum())}", flush=True)
    return pop, spike_count, active_steps

for g, nu in [(1.5, 4.0), (2.0, 4.0), (3.0, 4.0), (2.0, 8.0)]:
    print(f"\n===== g={g}, 背景 {nu} Hz =====")
    t0 = time.time()
    pop, sc, active = lif(g=g, nu_bg=nu)
    firing = sc / (4000*dt/1000.0)  # Hz per neuron
    firing_active = firing[firing > 0]
    late = pop[1000:]
    print(f"  群体放电率: 后期 {late.mean()*1000:.2f} Hz, 波动CV {late.std()/max(late.mean(),1e-9):.2f}")
    print(f"  活跃神经元 {len(firing_active):,} ({len(firing_active)/N:.1%}), 平均放电 {firing_active.mean():.1f} Hz")
    print(f"  自维持: {'是' if late[-500:].mean()*1000 > 1 else '否'}, 耗时 {time.time()-t0:.0f}s")
    np.save(f"pop_g{g}_nu{nu}.npy", pop)
