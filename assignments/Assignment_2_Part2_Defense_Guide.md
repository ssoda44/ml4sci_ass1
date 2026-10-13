# Assignment 2 Part 2：从头理解与答辩学习笔记

> 范围：Eindhoven Centraal 行人任务，以及当前 Assignment_2_part2.ipynb 的实现。
> 阅读顺序：①到⑧。每次学习一个模块；Defense checkpoint 先独立回答，再核对英文参考答案、中文解释与后续追问。
> 这是一份完整底稿，不要求一次学完。原来的 Core_Framework.md 保留为速查版。

## 0. 先建立知识地图

### 这部分在课程中的位置

课程先讨论如何用神经网络近似动力系统的演化，再讨论适合场和移动个体的结构，最后引入分布预测。Part 2 将这些内容结合起来：

**移动个体的历史 → 时间与图结构 → 条件 flow matching → 多种未来 → 实验诊断。**

- Lecture 1：surrogate modeling、历史到未来的 sequence-to-sequence 与 time bundling。
- Lecture 2：message passing、排列等变性与几何等变性的区别。
- Lecture 3：Lagrangian 系统、轨迹的 CNN 编码与图交互。
- Lecture 4：从随机起点到数据的 transport、flow matching 与生成。
- Practical 4：提供 conditional flow matching、bundling、prior 和 Euler 的实现骨架；当前实现重新设计时间处理与行人图。

一般 surrogate modeling 希望近似已有计算方法的行为。这个行人任务直接提供轨迹数据，没有要求我们对照某个特定行人求解器。因此，不能仅凭使用神经网络就声称已经证明加速了物理模拟。

### 优先级与八个模块

| 模块 | 必须掌握 | 应该理解的讨论角度 |
|---|---|---|
| ① 任务与预测对象 | 条件、目标、直接输出、完整生成结果 | 观测是否充分；概率模型的动机 |
| ② 图与空间交互 | 建图、边特征、消息与聚合 | 固定图限制；对称性；复杂度 |
| ③ 时间建模与 bundling | CNN 的时间轴、历史 context、共同处理未来 | 感受野；固定长度；与逐帧生成比较 |
| ④ Flow matching | 插值、监督方向、训练与生成的区别 | MSE 为什么不等同于预测平均未来 |
| ⑤ 两个 prior | 实际代码、随机性、条件独立、公平比较 | transport cost；理论动机与实验结论 |
| ⑥ 数据与实现 | 标准化、可变人数、场景级时间与 loss | 单位、权重、数据泄漏边界 |
| ⑦ 采样与位置恢复 | Euler、反标准化、速度积分 | 求解误差；两个时间轴；采样成本 |
| ⑧ 实验与结论 | 四个指标、四种分布、例子与结果 | oracle 指标、联合行为、可靠性与局限 |

**了解即可：** normalizing flow 的 Jacobian、Neural ODE 的 adjoint、EGNN 的详细构造、Eulerian PDE 网络和周期边界。本实现没有使用这些方法，不必把它们当作代码组成部分。

### 来源标记

- **代码事实**：当前 notebook 实际执行的操作。
- **课件/Practical 概念**：原始教学材料介绍的方法。
- **推导或扩展**：根据公式或实现进一步分析，不冒充课件原文。
- **待验证假设**：解释结果的可能机制，当前实验没有识别其因果作用。

### 统一符号

| 符号 | 含义 | 当前设置 |
|---|---|---|
| \(N_s\) | 场景 \(s\) 的行人数 | 数据中 5–100 |
| \(B\) | mini-batch 中场景数 | 通常 16 |
| \(P=\sum_sN_s\) | batch 的行人总数 | 随 batch 变化 |
| \(H\) | 观测帧数 | 10 |
| \(K\) | 未来帧数 | 20 |
| \(E\) | 一个场景的有向边数 | \(E\leq12N_s\) |
| \(d\) | 隐藏特征维度 | 64 |
| \(L\) | 未来 temporal–spatial block 数 | 3 |
| \(M\) | 每个场景生成的未来数量 | 20 |
| \(R\) | Euler flow 求解步数 | 30 |
| \(\Delta t\) | 数据的物理采样间隔 | 0.5 s |
| \(\tau\) | 人工 flow time | 0 到 1 |

本文用 \(p_i^t\) 表示物理位置，\(v_i^t\) 表示物理速度，\(X_\tau\) 表示标准化速度序列的 flow 状态，\(u_\theta\) 表示 flow vector field。

---

## ① 任务与预测对象

### 是什么、为什么需要概率预测

一个样本是整个场景的位置数组，形状为 \([N,30,2]\)。条件是第 0–9 帧，目标是第 10–29 帧的速度：

$$
C=\{p_i^t:t=0,\ldots,9\},\qquad
v_i^t=\frac{p_i^t-p_i^{t-1}}{\Delta t}.
$$

模型学习未来速度序列的条件分布：

$$
p_\theta(V^{10:29}\mid C).
$$

历史不直接提供目的地、意图等行为变量。相似的历史可能接上继续直走、转弯或减速等不同未来。这是概率建模的动机；它不等于已经证明数据具有 chaos。

**直观例子：** 两个人最近都向前走，但其中一人准备转向楼梯。只看位置历史，未必能区分两种意图。

### 三种“输出”必须区分

| 对象 | 形状 | 含义 |
|---|---|---|
| 网络直接输出 | \([P,20,2]\) | 当前序列应该如何随 flow time 变化 |
| flow 求解结果 | \([P,20,2]\) | 标准化的未来速度样本 |
| 完整生成器输出 | \([M,P,20,2]\) | 反标准化、积分后的未来位置 |

一次 draw 是整个场景的共同未来，包含所有行人。起始噪声独立，不意味着最终行人轨迹独立，因为网络通过图交换信息。

### 与 Part 1 的必要比较

| 方面 | Part 1：圆柱绕流 | Part 2：行人 |
|---|---|---|
| 表示 | 固定空间位置上的 Eulerian 流场 | 移动个体的 Lagrangian 轨迹 |
| 输入 | 当前速度场、网格几何、连接与边界类型 | 短期位置历史及导出特征 |
| 输出 | 1 秒后的速度场 | 未来 20 帧的序列 |
| 比较 | 确定性模型与概率模型 | 两种 prior 的概率模型 |

两部分的节点数量都可以在不同样本之间变化。不能说 Part 1 只有确定性预测，也不能把预测不确定性等同于底层物理必然随机。

### 答辩四层

- **Motivation：** 历史未完整描述未来行为所需的状态。
- **Technical details：** 学习条件下的整段、共同场景分布。
- **Implementation：** 条件只取前 10 帧，目标由位置差得到。
- **Results & implications：** 多个样本展示可能未来；是否合理仍需 evaluation。

### Defense checkpoint

1. What is the difference between the network output and one generated future? **【复习，covered】**
2. If every scene has only one recorded future, what assumptions allow learning a conditional distribution?
3. What experiment would test whether probabilistic prediction adds value over a deterministic baseline?

### Checkpoint 参考答案与理解

**Q1：预测对象与生成结果【covered，简短复习】**

> The network outputs a vector field for the current normalized future-velocity sequence. One generated future is obtained by integrating this field from flow time zero to one, denormalizing the resulting velocities, and accumulating them into positions. Each draw contains a joint future for the whole scene.

**代码对应：** forward 返回变化方向；sample_future 的 Euler 循环产生速度样本，再通过 denormalize_velocity 和 cumsum 得到位置。一次网络调用不等于一次完整生成。

**Q2：只有一个真实未来，怎样学习分布？**

> We share model parameters across many scenes and assume that related motion histories contain transferable statistical patterns. This allows the model to learn a conditional distribution by pooling information across examples. However, one recorded future per exact condition cannot identify that condition's full true distribution without assumptions about generalization.

**理解：** 依赖的是跨样本的统计共享、条件变化的规律与网络的 inductive bias，不是为同一条历史直接观察到 20 个真实未来。若模型逐个记住训练场景，它也可能只学到这些场景的单个终点；能生成多个样本不自动证明真实多模态已被学好。

**Q3：怎样检验概率模型有用？【未运行的实验方案】**

> I would train a deterministic bundled predictor using comparable temporal and graph components, the same splits, and a comparable training budget. I would compare pointwise accuracy, trajectory realism, joint interaction diagnostics, and sample-based distributional scores. A probabilistic model need not win average ADE to provide useful alternatives.

**理解：** Constant-velocity 可以作为便宜的 sanity baseline，但不能单独隔离“概率建模”的效果。较公平的确定性对照应尽量保留当前条件、时间和图处理，只改变预测目标与生成机制。当前只有两个概率模型，不能报告这个对照已经胜出。

### 老师继续追问：不确定性来自哪里？

**Question:** Is this aleatoric uncertainty or epistemic uncertainty?

> The random input is intended to represent alternative futures given incomplete observations. It is not a direct estimate of uncertainty about the learned weights. Our current model does not separately measure these two sources.

**扩展说明：** 条件未观测的意图与行为差异属于预测歧义；有限数据、模型选择和参数估计带来的未知属于模型不确定性。用不同训练 seed 比较稳定性可以提供后者的部分诊断，但不能把单个网络的采样 spread 自动解释为两者的完整分解。

---


## ② 图与空间交互

### 建图与边特征：代码事实

节点是行人。图第一次读取该场景时构建并缓存，只使用最后观测位置 \(p^9\)。

- 半径 3 m，最多保留 12 个最近入邻居。
- 排除自己；没有半径内邻居时，连接最近的另一人。
- \(N<2\) 时返回空图。
- 按接收节点分别选邻居，所以图是有向的，不保证每条边都有反向边。

对 \(j\rightarrow i\)：

$$
e_{ji}=[p_j^9-p_i^9,\ \|p_j^9-p_i^9\|,\ v_j^9-v_i^9]\in\mathbb R^5.
$$

例如，一个正在靠近的邻居与一个正在远离的邻居，即使距离相同，也可以具有不同相对速度。

### 如何计算交互

对每个未来帧 \(k\)，SpatialBlock 计算：

$$
m_{j\to i,k}=\phi([h_{i,k},h_{j,k},e_{ji}]),\qquad
a_{i,k}=\frac{1}{|\mathcal N(i)|}\sum_{j\in\mathcal N(i)}m_{j\to i,k}.
$$

再将自身特征与聚合消息输入 update MLP，使用残差与 LayerNorm 更新状态。空邻居时聚合为零，节点更新仍然执行。

**图和边特征固定，消息不固定。** 消息依赖当前隐藏状态，在每次 vector-field evaluation 中重新计算。三层交互支持多跳传播，但只能沿已有图的路径传播。

### 表达能力与代价

固定局部图减少无关的远距离消息，但未来新接近的行人可能没有直接连接。网络没有显式避碰力，也没有碰撞安全保证。

**推导：** 固定隐藏宽度时，每个 spatial block 的成本为 \(O(K(E+N))\)。因为 \(E\leq12N\)，该阶段近似为 \(O(KN)\)；全连接图为 \(O(KN^2)\)。

当前建图先计算全部 pairwise distances，因此其初始计算与距离矩阵存储仍有 \(O(N^2)\) 项。稀疏边没有把整个流程变成严格线性。

### 两个容易混淆的性质

共享消息函数与均值聚合，使消息层具有排列等变结构：同步重排行人和图后，输出应同步重排。普通 MLP 处理坐标向量并不自动提供旋转等变性；当前绝对位置还保留了固定车站的位置信息。

**扩展：** 验证随机模型的逐点排列等变性时，还要同步重排噪声。换了噪声样本，逐点输出不同不能直接说明缺乏等变性。

均值聚合控制消息尺度，但没有直接保留邻居数量。它可能从距离等特征获得部分密度线索，不能说全部密度信息都消失。

### 答辩四层

- **Motivation：** 表达邻近行人的影响。
- **Technical details：** 有向局部图、5 维边特征、均值消息聚合。
- **Implementation：** 缓存观测图；每次网络调用更新消息。
- **Results & implications：** 局部与多跳交互可学习，未来邻居变化仍是限制。

### Defense checkpoint

1. Why do fixed edges not imply fixed interaction messages? **【复习，covered】**
2. How would you test permutation equivariance in a stochastic model?
3. How would replacing mean aggregation with sum aggregation affect crowd-density information?

### Checkpoint 参考答案与理解

**Q1：固定图与变化的消息【covered，简短复习】**

> Only connectivity and observed edge features are fixed. At every vector-field evaluation, messages are recomputed from the current sender and receiver hidden states. Therefore, the strength and content of the learned interactions can change, even though the model cannot add new direct neighbours.

**代码对应：** SpatialBlock 每次调用 message MLP，再执行 index_add_ 和 degree 平均。三个 block 不共享各自的参数；每个 block 内对所有边共享同一个函数。

**Q2：怎样检验随机模型的排列等变性？**

> I would permute pedestrians within each scene and consistently permute all node data, prior noise, and edge indices. With fixed weights and matched numerical settings, I would compare the vector-field outputs or generated futures after undoing the permutation. Independent noise draws would test distributional consistency, not exact sample equality.

**实现要点：** 重排只在场景内进行，不混合不同场景。优先在固定图上检验网络结构；若同时重建 top-k 图，要注意距离并列时的选择规则与浮点误差。相同随机 seed 不一定让同一个行人获得相同噪声，需要明确重排噪声数组。

**Q3：mean 换成 sum 会怎样？**

> Sum aggregation makes the aggregate depend more directly on neighbour count and can retain density-related information. However, its magnitude can grow with degree, changing optimization and sensitivity to crowd size. Mean aggregation controls this scale but does not explicitly preserve neighbour count.

**理解：** 若收到的消息都接近同一个向量，mean 对 2 个与 12 个邻居可能相似，sum 会相差约 6 倍。mean 仍可能从几何与相对运动获得密度线索。可将 degree 作为附加特征作对照；degree 被 12 的上限截断，也不是无限密度的精确测量。

### 老师继续追问：为什么不动态更新图？

**Question:** Could you build the graph from predicted future positions?

> Yes, using predicted rather than ground-truth positions would not itself leak target information. It could capture changing neighbours, but noisy intermediate predictions may create unreliable edges. Hard neighbour selection can also make the field less smooth and increase computation.

**扩展说明：** 当前 flow 状态是候选速度序列，若据此恢复候选位置，早期噪声很大。每次为全部未来帧重建距离关系，还可能增加 \(O(RKN^2)\) 的图计算。软连接或近邻候选集是可讨论的替代，当前没有实现或验证。

---


## ③ 时间建模与 bundling

### CNN 沿哪一维处理

TemporalBlock 接收 \([P,T,d]\)，先转为 \([P,d,T]\)，再使用 Conv1d。时间是卷积的 length 轴；每个行人独立经过卷积，空间交互由 SpatialBlock 处理。

每个 block 有两个 kernel size 为 3 的卷积、SiLU、残差和 LayerNorm。dilation 扩大时间范围，例如一层 dilation=2 的卷积访问 \(t-2,t,t+2\)。

**例子：** 连续速度的变化有助于区分持续前进和减速；单个时刻不能直接表达这种时间变化。

### 历史编码与未来处理

历史每帧有 6 个特征：相对最后位置的位移 2 维、广播的最后绝对位置 2 维、速度 2 维。9 个观测速度通过重复第一个速度对齐到 10 个特征时刻。

~~~text
历史 [P,10,6]
→ Linear(6→64)
→ TemporalBlock(dilation=1), TemporalBlock(dilation=2)
→ 取最后隐藏状态 → context [P,64]

x_tau [P,20,2]
→ Linear(2→64) + context + future-frame embedding + flow-time embedding
→ [TemporalBlock → SpatialBlock] × 3
→ MLP head → u_theta [P,20,2]
~~~

未来三层 dilation 为 1、2、4。未来帧 embedding 标识第几个物理未来帧；flow-time embedding 标识当前生成阶段。这不是同一个时间。

Practical 4 的 bundled 示例将窗口展平后输入 MLP。展平仍能区分固定坐标槽位，但缺少 CNN 的显式时间局部性与沿时间共享滤波器的结构；不能说展平数学上彻底丢掉了全部顺序信息。

### Bundling 改变什么

整段 20 帧共同处理，避免在这段预测中逐帧反馈预测位置。生成仍要进行 30 次 flow 求解，不是一次网络调用就完成完整采样。

相对逐帧预测，它提供序列内的时间依赖；代价是更多序列特征存储。当前 learned frame embedding 与输出接口固定了 \(K=20\)，不能直接声称支持任意预测长度。

未来卷积访问相邻的候选未来状态是合法的：生成时这些状态来自 prior 与 flow 更新，不需要真实未来作为条件。

### 感受野边界：根据代码的推导

历史两个 block 各有两层卷积，总感受野跨度：

$$
R_{\mathrm{hist}}=1+2(3-1)(1+2)=13.
$$

最后输出索引为 9，直接特征范围为索引 3–9，另一侧落在 padding。因此“跨度大于 10”不等于最后状态完整使用全部 10 个特征时刻。索引 3 的速度差分还涉及位置索引 2。

未来三层 block 的感受野跨度为 29，但边界帧在单次调用中也不必直接看到全部 20 个时刻。不要把共同处理误解为每个输出都立即获得全局时间信息。

可以用梯度依赖检查感受野，用 pooling 对照实验检验历史压缩方式的影响；这些对照尚未作为正式结果运行。

### 答辩四层

- **Motivation：** 学习运动变化与整段未来的时间关系。
- **Technical details：** 沿时间卷积、dilation、两种时间编码。
- **Implementation：** 历史 context 与未来序列处理分开。
- **Results & implications：** 不作逐帧物理 rollout，仍受感受野和固定窗口限制。

### Defense checkpoint

1. Why does bundling still require multiple network evaluations? **【复习，covered】**
2. What would change if the future-frame embedding were removed?
3. How would you compare final-state history encoding with temporal pooling?

### Checkpoint 参考答案与理解

**Q1：bundling 为什么仍需多次调用？【covered，简短复习】**

> Bundling handles all twenty physical future frames together. Flow integration is a separate process: each Euler step evaluates how the entire sequence should change. We use thirty such steps to transform one prior draw into a future sample.

**理解：** Bundling 去掉这段预测中的逐帧物理反馈链，没有把 ODE 求解变成一步。

**Q2：去掉未来帧 embedding 会怎样？**

> The network would lose an explicit learned representation of the forecast horizon index. Temporal order would still be represented by the sequence layout and convolutions, with additional boundary effects. Removing the embedding does not make all frames identical, but could make horizon-specific behaviour harder to express.

**代码对应：** frame_embedding 的形状为 [20,64]，在未来状态投影后相加；flow-time embedding 不能替代它，因为所有未来帧可以同时处在同一个 \(\tau\)。是否改善结果要通过重新训练的 ablation 检验，当前没有该结果。

**Q3：last state 和 pooling 怎样比较？**

> I would retrain matched models that differ only in the history readout, comparing the final state with temporal pooling. Pooling can access features from every observed time step, while a final-state readout may emphasize recent motion. I would check both prediction quality and which input frames influence the context.

**理解：** 当前 last-state 的直接感受范围已在模块③推导。Mean pooling 也可能稀释近期特征；它不意味着输入时间顺序完全丢失，因为 pooling 之前已经经过 temporal convolution。可检查输入梯度，并做真正的 readout 对照训练。

### 老师继续追问：CNN 一定比展平 MLP 更好吗？

**Question:** Is a temporal CNN necessarily more expressive or cheaper than a flattened MLP?

> No. A flattened MLP can represent complex functions for a fixed window. The CNN supplies temporal locality and shared filters as an inductive bias. Whether this improves accuracy, parameter efficiency, or runtime depends on the architecture and must be measured.

**扩展说明：** CNN 的设计理由是让时间结构显式进入网络，不是证明 MLP 无法处理时间。当前没有 flat-MLP baseline，也没有计时结果，因此不能宣称已证实 CNN 更快或更准确。

---


## ④ Flow matching

### 核心任务：学习序列的变化率

用训练集统计量标准化真实未来物理速度，得到 \(X_1\)。抽取 prior \(X_0\) 与场景时间 \(\tau\sim U[0,1]\)：

$$
X_\tau=(1-\tau)X_0+\tau X_1,\qquad
u_{\mathrm{target}}=\frac{dX_\tau}{d\tau}=X_1-X_0.
$$

网络输入中间状态、flow time 与历史条件，输出同形状的向量场：

$$
u_\theta(X_\tau,\tau,C)\in\mathbb R^{P\times20\times2}.
$$

**例子：** 标量起点 2、终点 5，则 \(X_\tau=2+3\tau\)。终点为 5，变化率为 3，代码监督的是后者。

### 对应训练代码

~~~python
condition = model.encode_condition(batch)
physical_target = (batch.x[:, H:] - batch.x[:, H-1:-1]) / DT
x1 = model.normalize_velocity(physical_target)
x0 = draw_prior(model, condition, prior, generator)
tau = torch.rand(batch.num_graphs, device=batch.x.device, generator=generator)
tau_node = tau[batch.batch, None, None]
x_tau = (1 - tau_node) * x0 + tau_node * x1
prediction = model(x_tau, tau, condition)
loss = scene_balanced_loss(prediction, x1 - x0, batch.batch, batch.num_graphs)
~~~

同一场景的人共享 \(\tau\)，但不同场景可以使用不同 \(\tau\)。训练直接构造中间状态，不需要先求解 ODE；生成阶段才求解。

### 同样使用 MSE，为什么不直接变成平均未来

直接未来回归的最优 MSE 预测是 \(\mathbb E[X_1\mid C]\)。Flow matching 的回归对象是局部变化方向：

$$
u^*(x,\tau,C)
=\mathbb E[X_1-X_0\mid X_\tau=x,\tau,C].
$$

插值状态携带两端的信息。因此它平均给定状态与时间下的方向，而不是把所有最终轨迹合成一个终点。不同起点再沿学习到的场积分，可以产生不同未来。

这提供表示分布的机制，不保证有限网络已经学好了所有模式。训练 pair 的路径为直线，也不意味着生成轨迹必然是直线；生成使用的是学习到的状态相关向量场。

### 泄漏边界

Batch 保存全部 30 帧不自动构成泄漏。真实未来参与训练监督和插值；历史条件与图只使用观测。生成时不调用真实 \(X_1\)。

### 答辩四层

- **Motivation：** 学习分布的生成过程。
- **Technical details：** 线性插值与路径导数监督。
- **Implementation：** x_tau、tau、condition 输入同一个网络。
- **Results & implications：** 有多样性不等于模式覆盖或不确定性校准正确。

### Defense checkpoint

1. Why is the target x1 minus x0 rather than x1? **【复习，covered】**
2. What would happen if the network received x0 instead of x_tau during training?
3. Must sampled flow trajectories be straight if the training paths are straight?

### Checkpoint 参考答案与理解

**Q1：为什么是 x1-x0？【covered，简短复习】**

> The model predicts the derivative of the interpolation path with respect to flow time. Differentiating \((1-\tau)x_0+\tau x_1\) gives \(x_1-x_0\). The target is a rate of change of the sequence, not its endpoint.

**理解：** 改成监督 x1 属于另一种参数化，必须相应改变或转换采样方法；不能只换 target、继续把输出当导数积分。

**Q2：如果输入 x0 而不是 x_tau？**

> With conditionally independent endpoints, the optimal MSE predictor would be the conditional mean target minus x0. It would no longer learn the required field conditioned on intermediate flow states. Keeping the current sampler would therefore not implement the intended flow-matching transport and could contract samples toward the mean.

**数学扩展：**
$
\mathbb E[X_1-X_0\mid X_0=x_0,\tau,C]
=\mathbb E[X_1\mid C]-x_0.
$

这个目标缺少插值位置的信息。若理想网络学到 \(u(x,C)=\mu(C)-x\)，仍把当前状态送入现有连续 ODE，则：
$
X(1)=\mu(C)+e^{-1}(X_0-\mu(C)).
$
它是向均值收缩的分布，不能简单说必然立刻完全塌成一点。若始终保持起点输入不变，结果又不同。核心是训练输入与采样中的状态含义必须一致。

**Q3：生成路径一定是直线吗？**

> No. Each training pair defines a straight conditional path, but the learned field combines local directions from many possible endpoint pairs. Its direction can change with state and flow time, so integrating it can produce curved trajectories in flow space.

**理解：** 生成时不知道某个起点对应哪一个真实终点，不是在重放训练中指定的一条线。这里讨论的是高维速度序列空间中的路径，不是行人在车站中的物理路线。

### 老师继续追问：既然 MSE 有最优均值，为什么不会重复平均？

**Question:** What information prevents this from being ordinary mean-trajectory regression?

> The conditioning includes the intermediate state and flow time, not just the observed history. The averaged quantity is a local flow direction at that state. Fresh prior samples follow different state trajectories, so local averaging is compatible with a distribution of final outcomes.

**回答边界：** 原理支持分布表达，但本实验并未因此自动证明没有 mode collapse。模式覆盖仍须在生成结果中检验，不能只引用 loss 的形式。

---


## ⑤ Gaussian 与 informed prior

### 先定位代码中的差别

两种实验共用 PedestrianVectorField、flow_matching_loss、train_prior 和 sample_future，分别训练权重。差别在 prior_mean 返回的均值：

~~~python
# Uninformed
mean = observed.new_zeros((P, K, 2))

# Informed
recent = observed_velocities[:, -3:].mean(dim=1)
decay = GAMMA ** torch.arange(1, K + 1, ...)
physical_mean = recent[:, None] * decay[None, :, None]
mean = model.normalize_velocity(physical_mean)

# Both
x0 = mean + sigma * torch.randn(mean.shape, ...)
~~~

两种网络都接收历史。Informed 额外使用历史设置起点均值，本身仍然是 Gaussian。

### 定义与参数

$$
X_0^G=\sigma\epsilon,\qquad
X_{0,i,k}^I=\frac{\gamma^k\bar v_i-\mu_v}{s_v}+\sigma\epsilon_{i,k},
\quad \epsilon\sim\mathcal N(0,I).
$$

\(\bar v_i\) 是最近三个观测速度的平均，\(k=1,\ldots,20\)，两者 \(\sigma=1\)。全局 persistence 参数从训练场景拟合：

$$
\gamma=\operatorname{clip}_{[0,1]}
\frac{\sum_{\mathrm{train}}\sum_{i,t}v_i^{t+1}\cdot v_i^t}
{\sum_{\mathrm{train}}\sum_{i,t}\|v_i^t\|^2}.
$$

保存实验中 \(\gamma\approx0.9295\)，因此第 20 帧的均值保留约 23% 的近期速度。用最近三个速度平均可减小单次差分波动，也可能迟于突然的转向；全局一阶持续性外推不是每个行人的行为定律。

**实现细节：** 先在物理空间衰减，再标准化。Gaussian 的标准化零对应物理训练均值 \(\mu_v\)。若 \(\gamma=0\)，Informed 的物理均值为零，标准化均值为 \(-\mu_v/s_v\)，不一定与 Gaussian 完全相同。

### 条件独立与公平比较

$$
\pi_G(X_0,X_1\mid C)=p_0(X_0)p_1(X_1\mid C),\qquad
\pi_I(X_0,X_1\mid C)=p_0(X_0\mid C)p_1(X_1\mid C).
$$

两者在给定条件后独立抽 prior 与 target；Informed 不是根据当前真实未来挑选起点。近期速度只能来自观测，训练和生成使用相同 prior 规则。

保持架构、初始化种子、数据顺序、noise scale、20 epochs、优化器与 30 步 Euler 一致。两套模型最终权重不同；不是训练一套权重后直接换 prior。

### 理论动机与假设边界

Practical 的动机是减少 expected transport cost：

$$
\mathcal C=\mathbb E\|X_1-X_0\|^2.
$$

**数学扩展：** 条件独立、固定 noise scale、展平后维度为 \(D\) 时：

$$
\mathbb E[\|X_1-X_0\|^2\mid C]
=\operatorname{tr}\operatorname{Cov}(X_1\mid C)
+\|\mathbb E[X_1\mid C]-m(C)\|^2+D\sigma^2.
$$

均值接近真实条件均值可以减少该成本；当前衰减速度只是这一均值的近似。成本降低也不自动证明收敛更快或可用更少步数。所有 pair 原本就使用直线插值，不应把更短简单说成几何上更直。

当前没有正式测量 transport cost，也没有比较不同求解步数。急停、转弯与避让是 prior 可能不合适的情况。

### 答辩四层

- **Motivation：** 利用已有运动信息改善生成起点。
- **Technical details：** condition-dependent Gaussian mean，保持 noise scale。
- **Implementation：** 最近三次速度、训练拟合 gamma、物理衰减后标准化。
- **Results & implications：** 理论动机与实际优势分别判断。

### Defense checkpoint

1. Does zero mean in normalized velocity space mean stationary pedestrians? **【复习，covered】**
2. Does setting gamma to zero exactly recover our uninformed prior?
3. How would you test training efficiency and sampling efficiency separately from ADE?

### Checkpoint 参考答案与理解

**Q1：标准化零是否表示静止？【covered，简短复习】**

> No. Zero in standardized velocity space corresponds to the training-set mean physical velocity. The Gaussian prior therefore has physical mean \(\mu_v\), not necessarily zero. A physically stationary velocity would be normalized to \(-\mu_v/s_v\).

**代码对应：** normalize_velocity 减去 velocity_mean；denormalize_velocity 将其加回。当前 physical noise 的各坐标尺度为 \(\sigma s_v\)，不都是 1 m/s。

**Q2：gamma=0 是否等于 Gaussian？**

> Not exactly in our implementation. With gamma zero, the informed physical mean is zero, which becomes \(-\mu_v/s_v\) after normalization. The Gaussian mean is zero in normalized space. They coincide only when the training mean is zero, or if the informed mean definition is changed.

**理解：** Practical 中不带同样标准化背景的直观说法不能不加区分地搬过来。还要区分“noise scale 不变”和“均值完全相同”。

**Q3：怎样分别检验训练与采样效率？【未运行的实验方案】**

> For training, I would compare held-out prediction quality against training steps and wall-clock time across multiple seeds. For sampling, I would compare quality and runtime at several matched Euler step counts. ADE alone at twenty epochs and thirty sampling steps does not establish either efficiency advantage.

**理解：** 两种 prior 的 flow loss target 不同，不宜直接把更低的训练 loss 当作更快学会预测。最终 sampling 的网络调用成本主要相同；Informed 计算均值很便宜，但当前没有因此减少 R。

### 老师继续追问：没有提升，是否推翻理论？

**Question:** Does similar ADE prove that the informed prior is useless?

> No. The theory gives a conditional motivation, not a universal guarantee. Our results show no clear advantage in the tested displacement metrics. To explain why, we would need to measure transport cost and run prior ablations; the current experiment does not identify a causal mechanism.

**证据与假设分开：** 已验证的是表中小差异；“conditioning 已提供重复信息”“转弯使均值不合适”“衰减过快”仍是候选解释。初始均值更接近条件均值可减少成本，不保证有限训练和固定求解器下最终 ADE 更低。

---


## ⑥ 数据、归一化与可变人数

### 当前数据流程

从完整 100,000 个场景中随机选取 10,000 个，再按场景划分：

| seed | Train | Validation | Test |
|---|---:|---:|---:|
| 42 | 7,000 | 1,000 | 2,000 |

归一化与 gamma 只使用训练场景。验证用于选最佳 checkpoint，测试用于最终评价。

当前 velocity 统计使用训练场景全部相邻帧速度；position 统计使用观测位置。relative_std 实际是相对位移的 RMS 尺度，没有减去其均值，不能仅凭变量名当作普通标准差。

训练目标参与固定统计量拟合是允许的；不能用验证或测试来拟合这些量。

### 标准化改变了什么

$$
\tilde v=(v-\mu_v)/s_v,\qquad v=\tilde v\,s_v+\mu_v.
$$

Prior、target、插值与向量场训练都使用标准化空间；位置恢复使用物理空间。

归一化帮助控制数值尺度，但也改变误差权重。两个速度坐标按各自标准差缩放后，平方误差相当于对物理坐标误差按 \(1/s_v^2\) 加权。它不等同于直接优化米单位的 ADE。

### PyG batch 为什么不要求固定 N

两个场景分别有 3、5 人时：

~~~text
x.shape = [8,30,2]
batch.batch = [0,0,0,1,1,1,1,1]
~~~

Edges 的节点索引随场景偏移，不跨场景；用 batch.batch 将 \([B]\) 的 tau 映射到 \([P,1,1]\)，不能用固定人数 repeat。

### 场景等权 loss

令每个行人的坐标、帧平均误差为：

$$
\ell_{s,i}=\frac1{2K}\sum_{k=1}^K
\|u_{\theta,s,i,k}-(X_{1,s,i,k}-X_{0,s,i,k})\|^2.
$$

实际 loss：

$$
\mathcal L=\frac1B\sum_s\frac1{N_s}\sum_i\ell_{s,i}.
$$

100 人与 5 人场景获得相同总权重。直接对全部行人平均，会给予两个场景 20:1 的权重。场景等权与当前 pointwise evaluation 一致，但意味着小场景中单个行人的权重更高。

### 训练实现

AdamW，学习率 0.001，weight decay 0.0001，cosine schedule，梯度裁剪 1。训练 20 epochs，保存最低验证 loss 对应的权重。

验证每轮复用固定 noise/time，减少随机抽样造成的验证波动。两种 prior 的监督目标不同，不能只凭各自 flow loss 大小判断哪种最终预测更好。

### 答辩四层

- **Motivation：** 支持不同人数，控制尺度与评价权重。
- **Technical details：** 标准化、scene-balanced objective、场景时间广播。
- **Implementation：** PyG batch 索引与训练统计。
- **Results & implications：** 比较须保持数据与权重定义一致。

### Defense checkpoint

1. How does scene balancing change the contribution of crowded scenes? **【复习，covered】**
2. Does normalized MSE give equal weight to equal physical errors in both coordinates?
3. Why is fitting fixed statistics on training targets different from using test targets?

### Checkpoint 参考答案与理解

**Q1：场景等权的作用【covered，简短复习】**

> Scene balancing gives every scene equal total weight regardless of pedestrian count. It prevents a crowded scene from dominating solely because it contains more nodes. The trade-off is that an individual pedestrian in a small scene receives more weight than one in a large scene.

**例子：** 100 人与 5 人场景各占一半；不是 100 人场景占 20 倍。公平的定义取决于想评价“平均场景”还是“平均行人”，不是一个选项绝对正确。

**Q2：标准化后相同物理误差权重是否相同？**

> Not necessarily. Scaling each coordinate by its own standard deviation weights squared physical errors by the inverse variance of that coordinate. Equal errors in metres per second can therefore contribute differently to normalized MSE.

**推导：** 某坐标误差 \(\delta v_a\) 对 loss 的贡献与 \((\delta v_a/s_{v,a})^2\) 成正比。当前速度标准差约为 0.208、0.704 m/s，同样幅值的物理误差在前一个坐标获得约 \((0.704/0.208)^2\approx11.5\) 倍权重。共享坐标标准差或直接物理 loss 是可讨论的替代，未进行对照。

**Q3：训练目标统计与测试目标泄漏区别在哪里？**

> Training targets are allowed to fit fixed model parameters and preprocessing statistics. Those quantities are then frozen and reused at validation and inference. Using test targets to fit statistics or to construct a scene's prior would introduce information unavailable in a genuine held-out prediction.

**代码对应：** training_statistics 只接收 train_data，随后作为固定 buffers 使用。新场景的 Informed 均值只依赖观测；它不会拿该场景未来来重新估计 gamma。

### 老师继续追问：为何整场景共享 tau？

**Question:** Why do pedestrians in the same scene share one flow time?

> The model transports the entire scene sequence as one joint state. Sharing tau puts interacting pedestrians at the same stage of that transport. The batch mapping allows different scenes to use different times without assuming a fixed number of pedestrians.

**理解：** 场景内各坐标也使用同一个时间，物理未来帧并不各自抽不同 flow time。若给每个行人独立 tau，就改变了训练路径与场的定义，不能在现有公式下随意替换广播方式。

---


## ⑦ 采样与位置恢复

### 两个不同的时间轴

| 时间 | 意义 | 当前设置 |
|---|---|---|
| 物理时间 | 行人在车站中的运动 | 每步 0.5 s，20 步未来 |
| flow time | 把随机序列变成未来样本 | 0→1，30 步求解 |

10 个观测位置的首末跨度为 \(9\times0.5=4.5\) s；20 个未来位置相对最后观测位置覆盖 10 s。不要把帧数、采样间隔和首末跨度混用。

### Euler flow integration

~~~python
x = draw_prior(model, condition, prior, generator)
for step in range(steps):
    tau = x.new_full((num_scenes,), step / steps)
    x = x + model(x, tau, condition) / steps
~~~

$$
X_{\tau+\Delta\tau}=X_\tau+\Delta\tau\,u_\theta(X_\tau,\tau,C),
\qquad \Delta\tau=1/R.
$$

每次更新的是整个标准化速度序列，不是把一个行人向前移动一个物理帧。历史 context 编码一次并复用；模型设为 eval，关闭梯度。当前每批并行最多 4 个 draw，各自图通过索引偏移隔离。

### 从速度得到位置

$$
\hat v=\hat X_1s_v+\mu_v,\qquad
\hat p_i^{9+k}=p_i^9+\Delta t\sum_{j=1}^k\hat v_i^{9+j}.
$$

例如连续两步向右 \(1\) m/s，每步 0.5 s，位移为 1 m。遗漏 \(\Delta t\) 会把位移错误放大。

### 计算与误差的不同来源

固定宽度时，一次网络求解的主要未来处理成本约为 \(O(LK(E+N))\)，M 次抽样、R 步积分使其总成本约为 \(O(MRLK(E+N))\)，另有历史编码与缓存建图。

更大的 R 通常用于减少数值误差，但不保证对记录轨迹的 ADE 单调下降：更准确求解学到的场，与学到正确分布不是同一件事。

误差随物理未来增长，也可能来自累计速度误差与未来歧义。不能把增长曲线自动解释为逐帧 autoregressive feedback，本实现没有那条反馈链。

### 答辩四层

- **Motivation：** 从随机起点构造未来样本。
- **Technical details：** 人工时间 ODE 与物理速度积分。
- **Implementation：** Euler、条件缓存、draw 分批、DT=0.5。
- **Results & implications：** 数值误差、模型误差与预测不确定性需区分。

### Defense checkpoint

1. How do flow time and physical time differ? **【复习，covered】**
2. Would doubling Euler steps necessarily reduce ADE?
3. With a deterministic prior and deterministic inference, where would sample diversity come from?

### Checkpoint 参考答案与理解

**Q1：两个时间轴【covered，简短复习】**

> Physical time indexes pedestrian motion at half-second intervals. Flow time indexes the transformation of a complete candidate velocity sequence from prior to data. One Euler step changes that sequence in flow space; it does not advance pedestrians by half a physical second.

**代码对应：** /steps 是 flow 步长，DT*cumsum 是物理积分。不能互换两个系数。

**Q2：Euler 步数翻倍一定降低 ADE 吗？**

> No. More steps aim to reduce numerical error when solving the learned ODE. They do not correct a misspecified learned distribution. ADE against one recorded future can stay similar or even increase, despite a more accurate numerical solution.

**检验方式：** 固定权重、条件与初始噪声，比较 R=15、30、60 等设置。这样才能把数值求解变化与不同抽样分开；再汇总质量与耗时。该对照尚未运行。

**Q3：确定性 prior 怎么产生多样性？**

> With a deterministic prior and deterministic inference, the same condition produces the same future. In our current sampler, all meaningful randomness enters through the initial prior noise. Without it, there is no source of sample diversity.

**理解：** 训练时随机初始化的 embedding 在训练后是固定参数，不是每次抽样的新随机量。若使用 dropout、SDE 或其他噪声机制，就属于另一种设计；当前代码没有这些采样机制。

### 老师继续追问：20 个未来实际调用网络多少次？

**Question:** How many vector-field calls are made when sampling twenty futures?

> Each draw follows thirty flow steps. We batch four draws together, so twenty draws form five chunks. For one data batch, the sampler makes five times thirty, or 150, vector-field calls, each processing four scene-batch copies. This is equivalent to 600 single-draw field evaluations in workload, before accounting for batching and cached conditioning.

**代码对应：** draw_chunk_size=4，for start 控制 5 组，组内 for step 控制 30 次。不能把“并行得到多条轨迹”理解成计算量不随 M 增加，也不能仅凭调用次数就宣称 20 倍实测加速。

---


## ⑧ 实验、结果与能支持的结论

### 四个 pointwise 指标

对场景 s、draw m，位移误差为 \(d^t_{s,m,i}=\|\hat p^t_{s,m,i}-p^t_{s,i}\|_2\)：

$$
ADE_{s,m}=\frac1{N_sK}\sum_i\sum_{t=10}^{29}d^t_{s,m,i},
\qquad
FDE_{s,m}=\frac1{N_s}\sum_i d^{29}_{s,m,i}.
$$

ADE/FDE 对 draw 和场景平均。min20ADE/min20FDE 先在每个场景的 20 个完整 draw 中取最小，再对场景平均；二者可选到不同 draw。

不能为每个人独立选 best draw，那会拼成模型从未生成的场景。Best-of-20 是知道真实未来后的 oracle 指标，不能直接当作部署时的可选准确度。

### 四种 distributional quantity

| 量 | 定义或作用 | 当前实现边界 |
|---|---|---|
| Speed | \(\|v_i^t\|\) | 生成与真实未来的速度幅值 |
| Acceleration | \(\|(v_i^t-v_i^{t-1})/\Delta t\|\) | 首个未来加速度使用最后观测速度 |
| Tortuosity | 路长 / 净位移 | 包含 p9→p10，净位移下限 0.01 m |
| Closest neighbour | \(\min_{j\ne i}\|p_i^t-p_j^t\|\) | 使用全部同场景行人，不限于模型边 |

Pointwise 指标场景等权；histogram 按全部行人、帧、draw pooled，包含更多行人的场景贡献更多值。两种图表的权重逻辑不同。

### 当前保存结果

实验目录：part2_continued/e85b5013cdce；以下数值来自保存的 CSV/JSON，不是重新训练的结果。

| Prior | ADE (m) | FDE (m) | min20ADE (m) | min20FDE (m) |
|---|---:|---:|---:|---:|
| Gaussian | 0.9008 | 1.8254 | 0.6275 | 1.1916 |
| Informed | 0.9001 | 1.8265 | 0.6299 | 1.1929 |

**支持：** 本次配置下，informed mean 没有明显位移误差优势；20 次样本中存在比平均 draw 更接近记录未来的结果。

**不支持：** 两种生成分布完全相同；informed prior 永远无效；概率模型已经优于确定性 baseline；uncertainty 已校准。这里只有一个训练 seed，没有区间估计。

**待验证假设：** 历史 conditioning 已提供类似运动信息；近期速度对转弯/停止不充分；均值衰减削弱远期信息；两种 flow 都学到接近的最终分布。这些机制没有通过消融识别。

### 图表中的具体诊断

- Timewise ADE 从约 0.106 m 增至末端约 1.825 m；平均 20 个时间点得到 ADE，最后一点得到 FDE。
- Tortuosity 只展示 0–5；真实数据约 44.0% 在范围外，两种预测约 39%。仅看这一小段图会遗漏大量尾部。
- 最近距离图按题目显示 0–2 m。距离小于 0.2 m 的比例约为 GT 0.157%、Gaussian 0.876%、Informed 0.883%，提示近距离事件偏多。
- 近距离比例是 proxy，不是有身体几何定义的碰撞率。它不能自动证明所有这些事件都是碰撞。
- Histogram density 使用全部值作分母，所以裁剪范围里的面积可以小于 1；outside_fraction 报告遗漏质量，没有在显示范围内重新归一化。

### 例子与联合行为

六个例子按观测特征选择：随机、低速、高速、转向、近邻与拥挤。它们是诊断案例，不代表总体表现；不是按预测好坏挑选。图中的 10–90% 区间是 draw 分位数，不是统计置信区间。

速度等边缘分布匹配不充分证明交互正确。谁在什么时间避让谁、联合路线与事件时序，可能在 pooled histograms 中被隐藏。可以进一步检验共同接近、最小分离、时序关系或不确定性覆盖，但这些未作为当前正式结果实现。

### 答辩四层

- **Motivation：** 同时评价误差与生成行为。
- **Technical details：** 场景级 oracle 选择与 pooled 分布。
- **Implementation：** 20 draws、2000 scenes、观测边界衔接和完整场景距离。
- **Results & implications：** 无明显 prior 优势；部分行为统计仍需改善，机制尚未验证。

### Defense checkpoint

1. Why must best-of-20 selection be made for a complete scene? **【复习，covered】**
2. Can matching all four pooled histograms establish correct joint crowd dynamics?
3. What evidence would distinguish useful diversity from excessive or miscalibrated spread?

### Checkpoint 参考答案与理解

**Q1：为什么完整场景取 best-of-20？【covered，简短复习】**

> Each draw is one jointly generated crowd future. Selecting a different best draw for each pedestrian could combine mutually incompatible trajectories into a scene the model never generated. We therefore select the minimum scene-averaged error among complete draws.

**代码对应：** 每个 draw 先对所有行人平均，再执行 ade.min 或 fde.min。两种指标可以选到不同 draw，不代表存在同时达到两个 minimum 的同一个场景。

**Q2：四种 histogram 都匹配是否证明交互正确？**

> No. Pooled histograms summarize marginal statistics and discard pedestrian identity, event timing, and much of the temporal and joint structure. They can reveal discrepancies, but matching them does not establish correct coordinated behaviour or a correct joint conditional distribution.

**理解：** 最近距离比单人速度更接近交互诊断，但 pooled 最近距离也不能说明谁何时让路、是否持续避让或是否生成了不合理的联合选择。可增加同场景事件与时序评价，当前没有完整交互验证。

**Q3：怎样区分有用多样性与错误 spread？【扩展，未作为正式结果运行】**

> I would evaluate coverage together with sharpness and sample-based proper scores, using held-out observations. Wide intervals alone are not evidence of good uncertainty. I would also check trajectory plausibility and scene-level interaction failures, since diverse but unrealistic samples are not useful.

**理解：** Coverage 检查标称范围是否经常包含真实结果，sharpness 防止靠无限放宽范围获得覆盖，sample-based score 可以同时考虑接近真实结果与预测分布。可讨论 energy score；它不是本实现的 flow-training loss，也没有在当前 evaluation 中报告。每个精确条件只有一个真实未来，所以条件校准需要跨样本、分组与数据充分性假设，不能直接声称掌握完整真实分布。

### 老师继续追问：平均 ADE 较高是否说明生成模型更差？

**Question:** Can a realistic probabilistic model have worse average ADE than a point predictor?

> Yes. A sample can be plausible but differ from the particular recorded future. Averaging displacement errors over samples penalizes such alternatives, while a central point prediction can perform well under that metric. This is why we report both pointwise and behavioural diagnostics rather than judging distribution quality from ADE alone.

**回答边界：** 这是评价指标与分布预测的性质，不是本实验中“概率模型已优于确定性”的证据。min20 也不能单独补足校准或交互评价；更多 draw 可能改善 oracle minimum，却不改善分布本身。

---


## 9. 代码导航与学习方式

| 要理解什么 | 先读什么 |
|---|---|
| 数据选择与观测图 | LocalPedestriansDataset |
| 固定统计与 gamma | training_statistics |
| 已知条件 | encode_condition |
| 时间与空间层 | TemporalBlock、SpatialBlock、forward |
| 两个随机起点 | prior_mean、draw_prior |
| 训练目标与权重 | flow_matching_loss、scene_balanced_loss |
| 训练与验证 | train_prior |
| 从噪声到轨迹 | sample_future |
| 指标与分布 | trajectory_quantities、EvaluationAccumulator |

每个模块按同一顺序自检：先用自己的话解释 what/why，再定位 how 与公式，最后讨论假设、结果与边界。已标 covered 的问题只用于简短复习；新问题优先自己推理，不背标准答案。

---

## 10. 原始材料与出处

| 原始材料 | 本笔记使用的内容 |
|---|---|
| [Assignment 2](./Assignment_2.pdf) | 第 6–9 页：Part 2 的任务、两个 prior、架构与 evaluation；第 1–5 页：必要的 Part 1 对照 |
| [当前 Part 2 notebook](./Assignment_2_part2.ipynb) | 实际架构、统计、训练、采样、evaluation 与例子 |
| [当前 Part 1 notebook](./Assignment_2_part_1.ipynb) | 输入速度场、网格与边界信息，1 秒预测任务 |
| [Practical 4](<C:/Users/19670/Downloads/Practical4_Probabilistic_Lagrangian (1).ipynb>) | 单步与 bundled flow matching、展平 MLP 的限制、informed prior |
| [Lecture 1](C:/Users/19670/Downloads/course_files_export/01_Neural-surrogates.pdf) | 第 3、10、14–15 页：状态、surrogate、bundle 与 sequence-to-sequence |
| [Lecture 2](C:/Users/19670/Downloads/course_files_export/02_GDL_Eulerian_systems.pdf) | 第 33–34、38 页：message passing、排列与几何等变性 |
| [Lecture 3](C:/Users/19670/Downloads/course_files_export/03_Lagrangian-systems-v2.pdf) | 第 48、59 页：bundling 与轨迹 CNN/GNN |
| [Lecture 4](C:/Users/19670/Downloads/course_files_export/04_Probibilistic_neural_surrogates.pdf) | 第 49–50、53、55 页：路径、CFM、两种 MSE 回归与采样 |
| [Study Guide](<C:/Users/19670/Downloads/course_files_export/2AMM40_Study Guide.pdf>) | 第 2 页：答辩需要 motivation、technical details、implementation、results 与 implications |
| [当前指标](./part2_continued/e85b5013cdce/pointwise_metrics.csv) | 模块⑧的数值 |
| [Gaussian evaluation](./part2_continued/e85b5013cdce/gaussian_evaluation.json)、[Informed evaluation](./part2_continued/e85b5013cdce/informed_evaluation.json) | 时间误差、histogram 范围外比例、近距离事件比例 |
| [Training histories](./part2_continued/e85b5013cdce/gaussian_history.json)、[Informed history](./part2_continued/e85b5013cdce/informed_history.json) | 模块⑪的 training/validation loss 数值 |

课件没有证明本实现的性能，也没有证明行人数据的 uncertainty 校准或 chaos。感受野、复杂度与条件 transport cost 的细化是本笔记的推导；待验证机制不作为实验结论。

---

## 11. Training / validation loss 曲线：新补充的实验解读

[查看两种模型的 training / validation 曲线](./part2_continued/e85b5013cdce/training_validation_losses.png)。图已嵌入 notebook 的 Evaluation 部分；使用已有 history，没有重新训练。

### 实际数值

| Prior | Epoch 1 train | Epoch 1 val | Epoch 20 train | Epoch 20 val | 最低 val 所在 epoch |
|---|---:|---:|---:|---:|---:|
| Gaussian | 0.8733 | 0.7868 | 0.6671 | 0.6831 | 20 |
| Informed | 0.8595 | 0.7854 | 0.6691 | 0.6869 | 20 |

这些是标准化、场景等权的 flow-matching MSE，不是 ADE，也不是物理加速度误差。

### 曲线支持什么

前期下降较快，后期改善变小；两个模型的验证 loss 都总体下降，在这 20 epochs 内没有明显持续上升。末轮 val–train gap 约为 0.0159、0.0178。

**可说：** 这次训练没有明显发散，也没有从曲线看到持续的验证恶化。

**不能说：** 已完全收敛到最优解；不存在任何过拟合；Gaussian 在所有预测指标上更好；已经证明 prior 没有训练效率作用。最低 val 出现在最后一轮，也没有证明 20 epochs 是最佳训练长度。

### 计时记录的边界

History 的 seconds 可用于追踪单次运行，但不能直接当作严格的速度对照。首轮可能包含 GPU 初始化、缓存与 warm-up 等开销；公平计时应控制硬件、预热和重复测量。当前不能用首轮的耗时差证明 Informed 更快。

### 为什么第一轮 val 可以低于 train？

代码中的 train loss 是整轮内逐批记录的平均，包含较早、较差权重下的预测；val loss 在这一轮更新完成后计算。此外训练与验证场景和抽样分布不同，验证使用固定 noise/time。模型没有要求每一轮 train 必须低于 val。

### 可用的英文结果回答

> Both training and validation losses decrease, with rapid initial improvement followed by smaller changes. We do not observe a clear validation-loss upturn over twenty epochs. This supports stable optimization in this run, but does not prove optimal convergence or rule out generalization failures. Because the priors define different regression targets, their loss values alone do not determine which generates better futures.

**可能追问：** Is a small train–validation gap sufficient evidence of generalization?

> No. It is one diagnostic on the chosen split and objective. It does not establish performance on new environments, rare behaviours, or different crowd densities.

**理解：** 平均 loss 会隐藏小比例的急停、拥挤与近距离失败；这些需要分组或具体行为诊断。

---

## 12. 接下来怎样验证机制：实验设计，而不是已完成结果

下表都是**后续实验方案**。目前正式保存的证据仍是一个训练 seed、两个 prior、固定 20 epochs / 30 Euler steps 的结果。

| 要区分的问题 | 合适的对照 | 需要固定什么 | 不能从当前结果直接推出什么 |
|---|---|---|---|
| Informed 是否真的降低起点距离 | 在 validation 计算相同随机抽样预算下的 \(\mathbb E\|X_1-X_0\|^2\)，按未来帧分解 | 统计量、场景、noise scale | 无 ADE 优势不等于成本未下降 |
| 衰减是否合适 | 比较当前 gamma 与 gamma=1；gamma=0 要按物理零均值正确标注 | 历史平均窗口、网络、预算、sigma | gamma=0 不自动等同于 Gaussian |
| 近期平均是否帮助 | 比较最后 1 个与 3 个观测速度的均值 | 其余 prior 与训练设置 | 取平均没有保证比最新速度更好 |
| 图是否改善交互 | 重新训练保留空间消息与屏蔽空间消息的匹配模型，检查场景行为 | 时间结构、数据、预算、采样 | 仅存在 GNN 不证明因果改善 |
| history readout 是否限制信息 | 匹配训练 last-state 与 pooling 版本，先检查输入依赖 | 卷积层、宽度、其余模块 | 感受野限制不自动等于已导致误差 |
| 30 步是否足够 | 固定权重与初始噪声比较 R=15/30/60 的输出、质量与耗时 | 模型、场景、noise realization | 更多步数不保证更低 ADE |
| 小差异是否稳定 | 多训练 seed；对同场景的指标差做配对汇总与不确定性估计 | 各 seed 的公平训练与样本数 | 单个 seed 不证明统计显著或等价 |
| 是否有跨场景泛化 | 按速度、转向、人数分组；有来源 metadata 时检验独立时间/视频分组 | 分组定义与评价协议 | sample-level disjoint 不证明所有来源轨迹独立 |

### 对照实验不能只改 inference 然后声称公平训练比较

例如推理时直接删除边，可以测试训练后模型对边的敏感性，但也造成输入变化；这不是与“从头训练的无交互模型”相同的 ablation。改变 prior 的 gamma 时，也要区分只换 inference 起点的压力测试与重新训练的 prior 比较。

### 两种误差诊断的区别

- **固定权重，改变 R：** 主要检验求解与数值稳定性。
- **改变架构、prior 或 readout 并重新训练：** 检验建模与学习设计。
- **增加 M：** 检验采样量对估计稳定性及 oracle minimum 的影响，不代表每个 draw 更准确。

若所有设置都同时改变，不能把结果归因于其中一个因素。

### Validation 与 test 的用途

当前 test 已用于报告与诊断，不能拿它反复选择最优 gamma、半径、步数或网络。后续选择应基于 validation；若大量根据现有 test 结果改方案再挑赢家，应保留新的最终 holdout，或明确将结果定位为探索性诊断。

**可能追问：** If two metrics disagree, which one should decide the winner?

> There is no universal winner without a specified goal. Pointwise metrics measure agreement with the recorded future, while distributional and interaction diagnostics measure different aspects of generated behaviour. I would report the trade-off and choose validation criteria appropriate to the intended use.

**理解：** 不为某个 prior 临时改变评价标准，也不只报告它表现最好的指标。是否强调准确度、合理多样性或安全相关事件，属于任务目标的选择。

---

## 13. 更严格的答辩追问与回答边界

这些问题增加不同讨论维度，不再重复上面的 24 个 checkpoint。

### 13.1 为什么选半径 3 m、最多 12 个邻居？

> These are pragmatic choices that encode local interaction and bound the incoming degree. They are not established as optimal. A smaller radius may miss approaching pedestrians, while a larger radius or cap adds cost and potentially irrelevant messages. Their effect would need a validation ablation.

**事实边界：** 当前使用这组参数，未运行 radius/cap 系统对照。Nearest-neighbour fallback 还意味着图不完全遵守严格 3 m cutoff；它是额外的建图选择。

### 13.2 能否输入超过 100 个行人？

> The batching and graph code do not require a fixed pedestrian count, so the interface can accept larger scenes. However, our training data cover five to one hundred pedestrians. Supporting the tensor shape does not demonstrate predictive generalization, and the initial distance matrix still has quadratic cost.

**理解：** 可变 N 的工程支持、显存可承受性与超出训练人数范围的行为泛化，是三个不同问题。

### 13.3 可以用于另一个车站吗？

> We have not tested that. Local motion and relative interactions may transfer, but our absolute-position conditioning and training statistics encode the current environment. A new-station evaluation would be needed, with a clear distinction between zero-shot transfer and adaptation.

**事实边界：** 当前没有其他车站数据的结果。若重新拟合统计量或训练参数，应说明是 adaptation，不能仍称完全不调整的 zero-shot evaluation。当前位置特征也没有显式提供新车站的地图或出口。

### 13.4 随机场景划分是否保证完全独立？

> It prevents the same selected scene ID from appearing in both training and test. It does not by itself prove that different scenes come from independent videos, time intervals, or pedestrian tracks. Establishing that stronger claim requires source metadata that is not used by our current split.

**理解：** 当前按 assignment 要求做 sample-level split；未证实存在来源泄漏，也不能证实所有更强的独立性。不要把未检查的来源关系作为已发现的问题。

### 13.5 当前最值得关注的失败是什么？

> The nearest-neighbour statistics show more very close encounters in predictions than in ground truth. This is a useful interaction diagnostic, but it is not a collision rate because we have not defined pedestrian body geometry. I would inspect those events and use a physically justified criterion before making a safety claim.

**证据定位：** 具体比例见模块⑧，covered。它提示不足，不能用速度 histogram 大致匹配来抵消，也不能直接给模型“安全”结论。

### 13.6 Informed 没明显提升，工作还有什么意义？

> The experiment still tests a clear hypothesis under matched architecture, training, and sampling settings. It shows that this motion-informed mean does not provide a clear displacement advantage in the tested configuration. The contribution is an implementation aligned with temporal and spatial structure and a careful comparison, rather than a claim of universally improved prediction.

**边界：** “架构与任务一致”是设计理由，不等于当前实验证明它优于所有替代架构。结论可以有限但清楚；不需要把无明显优势包装成性能成功。

### 13.7 怎样解释一个例子，而不 cherry-pick？

> I would state how the scene was selected, distinguish a first draw from an oracle best draw, and compare the ensemble with the recorded future. I would use the example to diagnose behaviour, then rely on the full test-set metrics for population-level conclusions.

**当前事实：** 六种场景按观测特征选择；显示例子不按预测误差筛选。图上的 draw 分位区间不是模型性能的置信区间。端点 spread 是抽样变异，也不是端点误差。

---

## 14. 将理解变成可以讨论的答案

### 回答的组织方式

先明确问题问的是哪一层，然后回答这层并指向证据：

| 老师问的层次 | 应给出的内容 | 当前证据或边界 |
|---|---|---|
| Motivation | 为什么有限历史需要多种未来，为什么时间与图结构合理 | 任务条件与设计理由；不声称已证明 chaos |
| Technical details | 插值、条件向量场、prior、聚合与时间轴 | 模块①–⑦的公式 |
| Implementation | 数据切片、batch mapping、标准化、采样与 DT | 对应 notebook 函数 |
| Results & implications | 数值、图形现象、解释与替代解释 | 模块⑧与⑪；机制验证见⑫ |

不要把“为什么选这个方法”回答成完整背诵训练代码；也不要在解释曲线时只说“因为公式如此”。每个判断都说明是设计动机、代码事实、观察结果还是待验证假设。

### 被追问时的三个可用句式

- **有证据：** “In our saved run, we observe …; this supports … under these settings.”
- **没有因果验证：** “A possible explanation is …, but we have not isolated it with an ablation.”
- **方法边界：** “The implementation can represent …, but it does not enforce or establish ….”

这些是组织语言的工具，不是代替理解的标准答案。遇到 covered 问题，可简短回答并指出对应模块；需要展开时优先增加假设、验证方式或限制，不重新改写同一个观点。

### 下一轮自测

1. 不看笔记，画出条件、prior、vector field、Euler、反标准化和位置积分。
2. 在代码中找出只依赖历史的路径，与使用训练 target 的路径。
3. 选择一张 loss 图或轨迹图，分别说出它支持的一个结论和不能支持的一个结论。
4. 为一个失败机制设计单因素对照，说明应训练新模型还是只改变求解器。
