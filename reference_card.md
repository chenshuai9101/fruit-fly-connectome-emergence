# 连接组涌现基准卡 — 给 emergence-engine 的校准真值

> 数据：MaleCNS v1.0 雄性果蝇中枢神经系统连接组（169,984 神经元，25.6M 连接）
> 生成：`connectome_benchmark.py` → `reference_card.json`

## 核心结论（一句话）

**真实涌现系统（果蝇脑）的签名是「解剖模块化 + 谱上高秩 + 动力学临界」，而不是低秩 order parameter。** 你的引擎 `classify()` 会把真果蝇脑误判成 noise——这是个需要修的标定 bug。

---

## 度量对照表

| 连接组真值 | 引擎对应函数 | 校准判据 |
|---|---|---|
| **modularity Q = 0.44**（28 社区） | `criticality.modularity()` | 真涌现 ≈ **0.3–0.5**；~0 是 noise/collapse |
| **order_parameter = 0.0098** | `mode_spectrum.order_parameter()` | 大规模涌现 ≈ **~1%（不是 ~1.0，也不是 1/n）** |
| **participation_ratio = 3326** | `state()` 的 `participation_ratio` | 谱宽、非低秩 |
| **谱半径 ρ = 1.0**（列归一化） | （动力学临界性） | 涌现 = **临界 ρ≈1**；<1 静息、>1 爆发 |
| **度分布：非 scale-free**（top0.1% 仅 2.4% 边） | 结构先验 | hub 温和、非无标度 |
| **E/I 平衡 = 64% / 36%** | 角色分配 | 兴奋（proposer）/抑制（critic）平衡比 |

---

## 三个可操作的校准修正

### 修正 1：`classify()` 的 op 阈值是 scale 依赖的（关键 bug）

你 `mode_spectrum.classify()` 的判据：`op>0.8→collapse, op<0.25→noise, else→critical`。

- 这套阈值是为 **n≈4–20 个 agent** 调出来的（那时 1/n 是 0.05–0.25）。
- 真果蝇脑（n=17 万）的 op=0.0098，会被判成 **noise**——但它明明是自然界最典型的涌现系统。
- **修正**：大 N 时不要用 op 阈值判 emergence，改用 **modularity（Q≈0.4）+ 谱半径（ρ≈1）**。

### 修正 2：「涌现」的真正标志是模块度，不是低秩

果蝇脑解剖上高度模块化（Q=0.44、28 个脑区），但谱上高秩（需 15.2 万模式才覆盖 90% 方差）。

- 含义：**涌现 = 清晰的模块 + 宽谱内部结构**，而不是"少数模式奴役一切"（那是 collapse）。
- 你引擎的 `state()` 现在把 `participation_ratio` 当"Φ 参与度"，但对真实系统这个值应该**大**（高维参与），不是小。

### 修正 3：临界性要 ρ=1，不是"越复杂越好"

- 原始突触计数矩阵 ρ=3777（超临界爆发），列归一化后 ρ=1.0（临界）。
- 含义：涌现系统**处于临界边缘**。你引擎若要"临界态"判断，对标的是 ρ≈1 这个点，而不是某个绝对值。

---

## 结构先验（`connectome_prior.py`）

从真实数据抽的 block 结构，可给 agent 拓扑做模板：

- **28 个社区**（规模高度不均：视叶内在 ol_intrinsic 8.9 万 vs 最小的几百）
- **度分布**：非 scale-free，均值 145、中位 ~100、最大 ~1.1 万（温和重尾）
- **hub 集中在视觉系统**（ol_intrinsic、visual_centrifugal）
- **E/I 角色比 64/36**（按社区变化）

用法：`from connectome_prior import generate; A, comm, exc = generate(n_agents)` → 返回有向邻接 + 社区标签 + 兴奋标志。

---

## 如何用这张卡校准你的引擎

1. 跑一次 agent 模拟，用 `state()` 拿到 `{label, order_parameter, participation_ratio, ...}`。
2. 用 `criticality.modularity()` 算输出的模块度 Q。
3. 对照：
   - **Q ≈ 0.3–0.5** → 结构化了，涌现；
   - **Q ≈ 0** → 要么 collapse（全同）要么 noise（全异），都不是涌现；
   - 不要看 `label`（它会被 op 阈值骗），看 **Q + 谱半径**。
