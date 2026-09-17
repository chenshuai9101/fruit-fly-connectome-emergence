"""Connectome 涌现基准：用 emergence-engine 同一套度量公式，算真实果蝇脑的"真值"。

输出 reference_card.json + 可读摘要。度量对应关系：
  order_parameter   = 连接组 Wc 的 top 奇异值质量占比（引擎 mode_spectrum.order_parameter）
  participation_ratio = (Σs²)²/Σs⁴ 有效模式数（引擎 state() 的 participation_ratio）
  modularity        = Newman Q（解剖 superclass 为社区，引擎 criticality.modularity）
  谱半径 ρ          = 临界性（ρ=1 临界 / >1 超临界爆发 / <1 静息）
  complexity        = 有效模式数（引擎 C-H 平面的 C）
"""
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as sla, json, time

t0 = time.time()
d = np.load("graph.npz", allow_pickle=True)
N = int(d['N']); signs = d['signs']; superclass = d['superclass']
W = sp.csr_matrix((d['data'], d['indices'], d['indptr']), shape=(N, N))
W.eliminate_zeros()

card = {"N": N, "nnz": int(W.nnz), "total_synapse_weight": float(np.abs(W.data).sum())}

# ---- 1. 度分布 + 重尾 + hub ----
in_deg = np.diff(W.indptr).astype(np.int64)
out_deg = np.bincount(W.indices, minlength=N).astype(np.int64)
card["degree"] = {
    "in_mean": float(in_deg.mean()), "in_median": float(np.median(in_deg)), "in_max": int(in_deg.max()),
    "out_mean": float(out_deg.mean()), "out_median": float(np.median(out_deg)), "out_max": int(out_deg.max()),
}
# 重尾检查：出度 top-1%/0.1% 占比
out_sorted = np.sort(out_deg)[::-1]
top01 = int(np.ceil(N * 0.001))
card["degree"]["top0.1pct_out_share"] = float(out_sorted[:top01].sum() / max(out_deg.sum(), 1))
card["hubs"] = [{"idx": int(i), "out_deg": int(out_deg[i]), "superclass": str(superclass[i])}
                for i in np.argsort(out_deg)[::-1][:10]]

# ---- 2. Newman 模块度（superclass 社区，加权无向）----
uniq_sc = np.unique(superclass)
sc_map = {s: i for i, s in enumerate(uniq_sc)}
c = np.array([sc_map[s] for s in superclass])
M = float(np.abs(W.data).sum())          # 总突触权重（每有向边一次）
# within 计算：逐边判断 c[pre]==c[post]
pre_idx = np.repeat(np.arange(N), np.diff(W.indptr))
post_idx = W.indices
within = float(np.abs(W.data)[c[pre_idx] == c[post_idx]].sum())
in_strength = np.bincount(W.indices, weights=np.abs(W.data), minlength=N)
out_strength = np.bincount(pre_idx, weights=np.abs(W.data), minlength=N)
k = in_strength + out_strength          # 无向度
K_c = np.array([k[c == i].sum() for i in range(len(uniq_sc))])
Q = within / M - float(((K_c / (2 * M)) ** 2).sum())
card["modularity"] = {"Q_superclass": Q, "n_communities": int(len(uniq_sc))}

# ---- 3. 谱半径（临界性）----
def spectral_radius(A):
    v = np.random.randn(A.shape[0]); v /= np.linalg.norm(v); lr = []
    for _ in range(120):
        v = A.dot(v); n = np.linalg.norm(v)
        if n > 0:
            lr.append(np.log(n)); v /= n
    return float(np.exp(np.mean(lr[-50:]))) if lr else 0.0

rho_raw = spectral_radius(W)
col = np.bincount(W.indices, weights=np.abs(W.data), minlength=N).astype(float)
col[col == 0] = 1.0
Wc = W @ sp.diags(1.0 / col); Wc = Wc.tocsr()
rho_norm = spectral_radius(Wc)
card["spectral_radius"] = {"raw": rho_raw, "column_normalized": rho_norm}

# ---- 4. 模式谱（稀疏 svds top-k）+ order parameter + participation ratio ----
F2 = float((Wc.data ** 2).sum())        # ||Wc||_F² = Σ s²
print(f"[{time.time()-t0:.0f}s] 算 svds top-20 ...", flush=True)
try:
    s = sla.svds(Wc, k=20, which='LM', return_singular_vectors=False)
    s = np.sort(s)[::-1]
    s2 = s ** 2
    order_parameter = float(s2[0] / F2)          # top 模式质量占比（引擎 order_parameter）
    # participation_ratio = (Σs²)²/Σs⁴；Σs²=F2 精确，Σs⁴ = top20 + 尾部估计（尾部近似均匀）
    tail_mass = F2 - s2.sum()                    # 超出 top20 的方差
    n_tail = N - 20
    tail_s2_avg = tail_mass / n_tail             # 尾部平均 s²
    tail_s4 = n_tail * tail_s2_avg ** 2          # 尾部 Σs⁴ 估计（均匀假设，保守）
    sum_s4 = float((s2 ** 2).sum()) + tail_s4
    participation_ratio = float(F2 ** 2 / sum_s4)
    top20_var_share = float(s2.sum() / F2)
    # 有效模式数（覆盖 50%/90% 方差所需奇异值个数）
    def eff_modes(frac):
        cum = np.cumsum(s2)
        k = int(np.searchsorted(cum, frac * F2) + 1)
        if k <= 20:
            return k
        # 超出 top20：线性外推尾部
        return int(20 + (frac * F2 - cum[-1]) / tail_s2_avg)
    card["mode_spectrum"] = {
        "top_singular_values": s.tolist(),
        "order_parameter": order_parameter,
        "participation_ratio": participation_ratio,
        "top20_variance_share": top20_var_share,
        "effective_modes_50pct": eff_modes(0.5),
        "effective_modes_90pct": eff_modes(0.9),
    }
    print(f"  top-5 s: {s[:5].round(3)}")
    print(f"  order_param={order_parameter:.4f} (top模式质量占比)")
    print(f"  participation_ratio={participation_ratio:.1f} (有效模式数)")
    print(f"  top20 方差占比={top20_var_share:.3f}, 有效模式50%/90%={eff_modes(0.5)}/{eff_modes(0.9)}")
except Exception as e:
    print(f"  svds 失败: {e}")
    card["mode_spectrum"] = {"error": str(e)}

# ---- 5. E-I 平衡 ----
card["ei_balance"] = {
    "exc_neurons": int((signs > 0).sum()), "inh_neurons": int((signs < 0).sum()),
    "exc_edge_weight": float(np.abs(W.data[signs[W.indices] > 0]).sum()),
    "inh_edge_weight": float(np.abs(W.data[signs[W.indices] < 0]).sum()),
}

json.dump(card, open("reference_card.json", "w"), indent=2, ensure_ascii=False)
print(f"\n完成，总耗时 {time.time()-t0:.0f}s。已存 reference_card.json")
print("\n===== 可读摘要 =====")
print(json.dumps(card, indent=2, ensure_ascii=False))
