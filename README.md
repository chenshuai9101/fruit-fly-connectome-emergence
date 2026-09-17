# Fruit Fly Connectome Emergence（果蝇连接组涌现）

> **What:** 用完整雄性果蝇中枢神经系统连接组（MaleCNS v1.0）做「涌现」分析——抽取真实生物涌现系统的**度量真值**，做成多智能体涌现引擎的**校准基准 + 结构先验**。
>
> **Why it matters:** 一个真实的涌现系统（17 万神经元）的度量签名，被用来发现并修复自建涌现引擎的一个尺度相关 bug。

## 一句话结论

果蝇脑的涌现签名是 **「解剖模块化（Q=0.44）+ 谱上高秩 + 动力学临界（ρ=1）」**，而不是低秩 order parameter。这条签名暴露了 `emergence-engine` 的 `classify()` 用固定阈值会把大规模结构化系统误判成 noise。

## 关键发现

| 度量 | 值 | 含义 |
|---|---|---|
| 神经元 / 连接 | 169,984 / 25.6M（总突触权重 1.21 亿） | 完整 CNS 连接组 |
| Newman 模块度 Q | **0.44**（28 个解剖脑区） | 解剖上高度模块化 |
| order_parameter | **0.0098** | top 模式仅占 1% 方差 → 谱上**高秩** |
| participation_ratio | **3326**（90% 方差需 15.2 万模式） | 非低维、近噪声的宽谱 |
| 谱半径 ρ | 原始 **3777** / 列归一化 **1.0** | 原始矩阵超临界爆发；归一化后精确临界 |
| 度分布 | 均值 145、最大 ~1.1 万；**非 scale-free**（top 0.1% 仅 2.4% 边） | 温和重尾，hub 集中在视觉系统 |
| E/I 平衡 | 兴奋 103,783 / 抑制 57,343（**64/36**） | 按突触前递质（ACh 兴奋 / GABA·Glu·His 抑制） |

## 仓库结构

```
build_graph.py            # 下载的 feather 数据 → 带符号 CSR 连接矩阵 (graph.npz)
connectome_benchmark.py   # 抽取涌现真值 → reference_card.json
connectome_prior.py       # 结构先验生成器（SBM 块模型：28 社区 + 非 scale-free 度分布 + 64/36 E/I）
run_dynamics*.py          # Wilson-Cowan 速率动力学（原始 ρ≈3777 饱和 / 列归一化 ρ=1 不动点）
run_lif*.py / run_final.py# LIF 脉冲动力学（行归一化 + 背景驱动 → 自维持放电；循环 ON/OFF 对照）
reference_card.md         # 校准解读文档（引擎度量对照 + 3 个校准修正）
reference_card.json       # 涌现真值（机器可读）
```

## 复现

1. **数据**（`connectome-weights` 1.05GB 即可，`syn-partners` 6.7GB 可选）：

   ```
   gs://flyem-male-cns/v1.0/connectome-data/flat-connectome/connectome-weights-male-cns-v1.0-minconf-0.5.feather
   ```

   另需 `body-annotations-...minconf-0.5.feather` + `body-neurotransmitters-male-cns-v1.0.feather`。

2. **依赖**：`pyarrow`（读 feather）+ `numpy/scipy`。数据下载需清本地代理 `env -u HTTPS_PROXY ...`（本机 claude-proxy 的 8081 MITM stub）。

3. **跑法**：

   ```bash
   python3 build_graph.py          # 建图 → graph.npz（~28s）
   python3 connectome_benchmark.py # 抽真值 → reference_card.json（~8s）
   python3 connectome_prior.py     # 生成连接组式 agent 图（SBM 先验）
   ```

## 落地：校准 emergence-engine

真值暴露了自建涌现引擎 `mode_spectrum.classify()` 的 **尺度依赖 bug**（`op<0.25→noise` 是为 n≈4–20 调的，会误判大 N 结构化系统）。修复：

- `classify()` 改用尺度无关的 `participation_ratio`（`pr<2→collapse, pr>0.6n→noise`）
- `state()` 新增 `modularity` 字段（Newman Q）作为规模无关的涌现主判据

校准判据：**agent 输出的 `modularity ≈ 0.3–0.5` = 结构化涌现（≈果蝇脑 0.44）**；不要只看 `label`。

> 引擎代码在本地 `~/emergence-engine/`（改动：`mode_spectrum.py`、`engine_unified.py`）。

## 数据与许可

连接组数据来自 MaleCNS v1.0（Google Research / HHMI Janelia / Cambridge / MRC），**CC-BY** 协议开源，详见 <https://male-cns.janelia.org>。
