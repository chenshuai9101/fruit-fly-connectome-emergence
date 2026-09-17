"""Final: control experiment (recurrent ON/OFF) + functional clustering vs anatomy."""
import numpy as np, scipy.sparse as sp, time

d = np.load("graph.npz", allow_pickle=True)
N = int(d['N']); signs = d['signs']; superclass = d['superclass']
W = sp.csr_matrix((d['data'], d['indices'], d['indptr']), shape=(N, N))
W.eliminate_zeros()
row_sum = np.bincount(np.repeat(np.arange(N), np.diff(W.indptr)),
                      weights=np.abs(W.data), minlength=N).astype(np.float64)
row_sum[row_sum == 0] = 1.0
Wr = sp.diags(1.0/row_sum) @ W; Wr = Wr.tocsr()

dt = 0.5; tau = 20.0; Vth = 1.0; ref_steps = 4

def lif(Wrec, g, mu, sig, T, raster_ids=None, seed=0):
    rng = np.random.default_rng(seed)
    V = np.zeros(N); ref = np.zeros(N, dtype=np.int32)
    pop = np.zeros(T); spike_count = np.zeros(N)
    raster = np.zeros((len(raster_ids), T), dtype=np.uint8) if raster_ids is not None else None
    for t in range(T):
        spk = V >= Vth
        I = g * Wrec.dot(spk.astype(np.float64))
        mask = ref == 0
        V[mask] += (dt/tau) * (-V[mask] + mu) + I[mask]
        V[mask] += sig * rng.standard_normal(int(mask.sum()))
        V[spk] = 0.0; ref[spk] = ref_steps
        ref = np.maximum(ref - 1, 0)
        spike_count += spk; pop[t] = spk.mean()
        if raster is not None and spk.any():
            raster[:, t] = spk[raster_ids]
    return pop, spike_count, raster

g, mu, sig, T = 2.0, 0.8, 0.12, 4000

# ---- control: recurrent OFF ----
print("===== 对照: 无循环连接 (Wr=0) =====")
t0 = time.time()
pop_off, sc_off, _ = lif(sp.csr_matrix((N, N)), g, mu, sig, T)
fr_off = pop_off[1000:].mean() * 1000
print(f"  群体放电率 {fr_off:.2f} Hz, 活跃神经元 {int((sc_off>0).sum()):,}")

# ---- recurrent ON + raster for 3000 sampled neurons ----
print("\n===== 循环连接 ON + 记录 3000 神经元 raster =====")
rng = np.random.default_rng(1)
raster_ids = rng.choice(N, 3000, replace=False)
t0 = time.time()
pop_on, sc_on, raster = lif(Wr, g, mu, sig, T, raster_ids)
fr_on = pop_on[1000:].mean() * 1000
print(f"  群体放电率 {fr_on:.2f} Hz, 活跃神经元 {int((sc_on>0).sum()):,}")

print(f"\n>>> 循环结构净效应: {fr_on:.2f} Hz (ON) vs {fr_off:.2f} Hz (OFF), 差 {fr_on-fr_off:+.2f} Hz")

# ---- firing rate distribution ----
firing = sc_on / (T*dt/1000.0)
fa = firing[firing > 0]
print(f"\n放电率分布: 中位 {np.median(firing):.1f} Hz, 90分位 {np.percentile(firing,90):.1f} Hz, "
      f"最大 {firing.max():.1f} Hz, 偏态 {float(np.mean(fa>3*np.median(fa))):.3f}")

# ---- functional clustering vs anatomy ----
print("\n===== 功能聚类 vs 解剖 (superclass) =====")
binw = 20  # 10ms bins
bins = raster.reshape(len(raster_ids), T//binw, binw).sum(axis=2).astype(np.float64)  # 3000 x 200
bins -= bins.mean(axis=1, keepdims=True)
C = np.corrcoef(bins)  # 3000x3000
sc_ids = superclass[raster_ids]
same = np.zeros(len(raster_ids)); cross = np.zeros(len(raster_ids))
# 稀疏比较: 每个神经元取与它同类的近邻 vs 异类
from collections import Counter
cnt = Counter(sc_ids)
i, j = np.triu_indices(len(raster_ids), 1)
is_same = sc_ids[i] == sc_ids[j]
same_mask = is_same; cross_mask = ~is_same
print(f"  同类神经元对相关均值: {C[i[same_mask], j[same_mask]].mean():.4f}  (n={same_mask.sum():,})")
print(f"  异类神经元对相关均值: {C[i[cross_mask], j[cross_mask]].mean():.4f}  (n={cross_mask.sum():,})")
np.save("pop_on.npy", pop_on); np.save("pop_off.npy", pop_off)
np.save("raster_ids.npy", raster_ids); np.save("sc_ids_sample.npy", sc_ids)
np.save("corr_C.npy", C)
print("\n完成")
