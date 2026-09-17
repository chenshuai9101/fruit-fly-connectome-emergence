"""结构先验生成器：从真实连接组抽 block 结构，生成"连接组式"的 agent 交互图。

用途：给 emergence-engine 的 agent 拓扑做模板（当前引擎无拓扑概念，这是前瞻 + 可复用的结构先验）。
方法：随机块模型 (SBM)——社区规模、块间连接概率、度分布、E/I 角色比例全部取自真实数据。
"""
import numpy as np, scipy.sparse as sp, json

def load_block_structure(npz="graph.npz"):
    """从真实连接组抽：社区规模、块连接矩阵、每社区兴奋比例、度分布。"""
    d = np.load(npz, allow_pickle=True)
    N = int(d['N']); superclass = d['superclass']; signs = d['signs']
    W = sp.csr_matrix((d['data'], d['indices'], d['indptr']), shape=(N, N))
    W.eliminate_zeros()

    uniq = np.unique(superclass); c = {s: i for i, s in enumerate(uniq)}
    ci = np.array([c[s] for s in superclass])
    sizes = np.array([int((ci == i).sum()) for i in range(len(uniq))])

    # 块连接矩阵：B[i,j] = 社区 i(前) -> 社区 j(后) 的边数
    pre = np.repeat(np.arange(N), np.diff(W.indptr))
    post = W.indices
    B = np.zeros((len(uniq), len(uniq)))
    np.add.at(B, (ci[pre], ci[post]), 1)

    # 每社区兴奋（ACh）神经元比例
    exc_frac = np.array([float((signs[ci == i] > 0).mean()) for i in range(len(uniq))])

    # 度分布（出度，采样用）
    out_deg = np.bincount(W.indices, minlength=N)
    return {"sizes": sizes, "B": B, "exc_frac": exc_frac, "out_deg": out_deg,
            "n_comm": len(uniq), "N": N, "names": uniq}


def generate(n_agents, block=None, seed=0):
    """生成 n_agents 个 agent 的交互图（有向），返回 (adjacency csr, community, exc_flag)。

    block=None 时自动从真实数据加载结构（社区规模按比例缩放）。
    """
    rng = np.random.default_rng(seed)
    if block is None:
        block = load_block_structure()
    sizes = block["sizes"]; B = block["B"]; exc_frac = block["exc_frac"]
    N0 = block["N"]; out_deg = block["out_deg"]

    # 1. 分配社区（按真实规模比例）
    p = sizes / sizes.sum()
    comm = rng.choice(len(sizes), size=n_agents, p=p)

    # 2. 分配兴奋/抑制角色（按社区比例）
    exc = np.zeros(n_agents, dtype=bool)
    for k in range(len(sizes)):
        m = comm == k
        exc[m] = rng.random(int(m.sum())) < exc_frac[k]

    # 3. 采样度（直接取真实度分布——度是节点绝对属性，保留 hub 结构）
    deg_sample = out_deg[rng.choice(N0, size=n_agents, replace=True)]
    deg = np.maximum(1, deg_sample.astype(int))

    # 4. 按块连接概率连线：每条边按 B 选目标社区，社区内按度加权选节点
    P = B / B.sum()          # 全局 (src_comm, dst_comm) 概率
    rows, cols = [], []
    node_idx_by_comm = [np.where(comm == k)[0] for k in range(len(sizes))]
    # 目标节点按"入度倾向"加权：用该节点真实出度归一化（简化：均匀）
    for a in range(n_agents):
        src_comm = comm[a]
        k = deg[a]
        row = P[src_comm]
        p_dst = row / row.sum() if row.sum() > 0 else np.full(len(sizes), 1.0 / len(sizes))
        dst_comm = rng.choice(len(sizes), size=k, p=p_dst)
        for dc in dst_comm:
            cands = node_idx_by_comm[dc]
            if len(cands):
                rows.append(a); cols.append(rng.choice(cands))
    if rows:
        A = sp.csr_matrix((np.ones(len(rows)), (np.array(rows), np.array(cols))),
                          shape=(n_agents, n_agents))
    else:
        A = sp.csr_matrix((n_agents, n_agents))
    return A, comm, exc


if __name__ == "__main__":
    blk = load_block_structure()
    print(f"真实连接组 block 结构: {blk['n_comm']} 社区, N={blk['N']}")
    print(f"社区规模 top8: {sorted(zip(blk['names'], blk['sizes']), key=lambda x:-x[1])[:8]}")
    # 生成一个 1000 agent 的图验证
    A, comm, exc = generate(1000, block=blk, seed=0)
    n = A.shape[0]; m = A.nnz
    print(f"\n生成 1000-agent 图: {m} 边, 平均度 {m/n:.1f}")
    # 简单验证：计算模块度（社区标签）
    M = m
    within = sum(1 for i in range(n) for j in A[i].indices if comm[i] == comm[j])
    k = np.diff(A.indptr)
    Q = within / M - sum((k[comm == c].sum() / M) ** 2 for c in np.unique(comm))
    print(f"生成图模块度 Q ≈ {Q:.3f} (真实 0.44)")
