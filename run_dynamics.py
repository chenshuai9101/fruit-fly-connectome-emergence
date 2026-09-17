"""Run rate (Wilson-Cowan) dynamics on the real MaleCNS connectome and check for emergent activity."""
import numpy as np, scipy.sparse as sp, time, sys

# ---------- load ----------
d = np.load("graph.npz", allow_pickle=True)
indptr, indices, data = d['indptr'], d['indices'], d['data']
N = int(d['N']); signs = d['signs']; nt = d['nt']; superclass = d['superclass']
W = sp.csr_matrix((data, indices, indptr), shape=(N, N))
W.eliminate_zeros()
print(f"神经元 N={N:,}  nnz={W.nnz:,}")

# ---------- degree stats ----------
in_deg = np.diff(W.indptr).astype(np.float64)           # 每神经元突触前伙伴数
out_deg = np.bincount(W.indices, minlength=N).astype(np.float64)  # 每神经元突触后伙伴数
print(f"平均入度 {in_deg.mean():.1f}, 中位 {np.median(in_deg):.0f}; 平均出度 {out_deg.mean():.1f}")
print(f"兴奋神经元 {int((signs>0).sum()):,} / 抑制 {int((signs<0).sum()):,}")

# ---------- spectral radius via power iteration ----------
v = np.random.randn(N).astype(np.float64); v /= np.linalg.norm(v)
logrho = []
for k in range(120):
    v = W.dot(v)
    n = np.linalg.norm(v)
    if n == 0: print("W 零矩阵，无自发动力学"); sys.exit()
    if k > 20: logrho.append(np.log(n))
    v /= n
rho = float(np.exp(np.mean(logrho[-60:])))
print(f"谱半径 ρ(W) ≈ {rho:.3f}")

# ---------- normalize: scale W so ρ ~ gain target ----------
# 用全局增益 g；W 本身保留相对结构
# ---------- Wilson-Cowan 速率动力学 ----------
def run(g, tau=10.0, dt=0.1, T=4000, sigma=0.05, I_ext=0.0, kick=1.0, kick_until=500):
    r = np.zeros(N, dtype=np.float64)
    mean_trace = np.zeros(T)
    exc = signs > 0; inh = signs < 0
    for t in range(T):
        x = g * W.dot(r) + I_ext + sigma * np.random.randn(N)
        if t < kick_until:
            x = x + kick * 0.02 * np.random.randn(N)   # 短暂外驱动 kick 启动
        r = r + dt/tau * (-r + np.tanh(x))
        # 记录群体均值
        mean_trace[t] = r.mean()
        if (t+1) % 1000 == 0:
            print(f"  g={g}: t={t+1}, mean_act={r.mean():.4f}, max={r.max():.3f}, "
                  f"exc={r[exc].mean():.4f} inh={r[inh].mean():.4f}", flush=True)
    return r, mean_trace, exc, inh

for g in [0.5, 1.0, 2.0, 4.0]:
    print(f"\n===== 增益 g = {g} =====")
    t0 = time.time()
    r, trace, exc, inh = run(g)
    late = trace[-1000:].mean()
    early = trace[:500].mean()
    print(f"  早期均值 {early:.4f} -> 后期均值 {late:.4f}  "
          f"({'自维持' if late > 0.1 else '衰减/静息'}), 耗时 {time.time()-t0:.0f}s")
    print(f"  最终兴奋群体 {r[exc].mean():.4f}, 抑制群体 {r[inh].mean():.4f}")
    np.save(f"trace_g{g}.npy", trace)
