"""Column-normalized (conserved output) dynamics -> critical regime. Look for emergent activity."""
import numpy as np, scipy.sparse as sp, time

d = np.load("graph.npz", allow_pickle=True)
N = int(d['N']); signs = d['signs']
W = sp.csr_matrix((d['data'], d['indices'], d['indptr']), shape=(N, N))
W.eliminate_zeros()
print(f"N={N:,} nnz={W.nnz:,}")

# ---- column normalize: each neuron's total |output| strength = 1 ----
col_strength = np.bincount(W.indices, weights=np.abs(W.data), minlength=N).astype(np.float64)
scale = np.zeros(N); scale[col_strength > 0] = 1.0 / col_strength[col_strength > 0]
Wc = W @ sp.diags(scale)          # column j scaled by 1/out_strength[j]
Wc = Wc.tocsr()

# spectral radius of Wc (power iteration)
v = np.random.randn(N); v /= np.linalg.norm(v); lr = []
for k in range(120):
    v = Wc.dot(v); n = np.linalg.norm(v)
    if n > 0:
        if k > 20: lr.append(np.log(n))
        v /= n
rho = float(np.exp(np.mean(lr[-60:]))) if lr else 0
print(f"列归一化后谱半径 ρ(Wc) ≈ {rho:.4f}")

exc = signs > 0; inh = signs < 0
print(f"兴奋 {int(exc.sum()):,} / 抑制 {int(inh.sum()):,}")

def simulate(drive=0.0, sigma=0.05, tau=1.0, dt=0.05, T=6000, drive_frac=0.05):
    r = np.zeros(N)
    # 对一小撮神经元施加恒定驱动(模拟自发背景输入)
    drive_ids = np.random.choice(N, int(N*drive_frac), replace=False)
    ext = np.zeros(N); ext[drive_ids] = drive
    trace = np.zeros(T); r0 = np.zeros(T)
    rng = np.random.default_rng(0)
    for t in range(T):
        x = Wc.dot(r) + ext + sigma * rng.standard_normal(N)
        r = r + dt/tau * (-r + np.tanh(x))
        trace[t] = r.mean(); r0[t] = float(np.sqrt((r*r).mean()))
    return r, trace, r0, drive_ids

for drive in [0.0, 0.5, 1.0]:
    print(f"\n===== 驱动 drive={drive} (施加于5%神经元) =====")
    t0=time.time()
    r, trace, r0, dids = simulate(drive=drive)
    late = trace[-2000:];
    print(f"  群体均值: 后期 {late.mean():.4f} ± {late.std():.4f}, 全期波动 {trace.std():.4f}, 耗时 {time.time()-t0:.0f}s")
    print(f"  RMS活动 {r0[-1]:.4f}, 兴奋均值 {r[exc].mean():.4f}, 抑制均值 {r[inh].mean():.4f}")
    # 单神经元活动分布
    h, e = np.histogram(r, bins=50)
    # 活动是否自发持续: 去掉外部驱动后仍否
    print(f"  活跃神经元占比(|r|>0.1): {float((np.abs(r)>0.1).mean()):.3f}, (|r|>0.5): {float((np.abs(r)>0.5).mean()):.3f}")
    np.save(f"trace2_d{drive}.npy", trace); np.save(f"r_final_d{drive}.npy", r)
