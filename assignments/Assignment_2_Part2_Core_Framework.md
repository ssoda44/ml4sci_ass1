# Assignment 2 Part 2：核心框架与 idea

## 1. 核心 idea

根据行人过去的运动与周围人的状态，生成整个场景的多种可能未来。使用 **Temporal CNN + Graph message passing** 处理时间关系与行人交互，通过 **conditional flow matching** 将随机速度序列变成合理的未来速度序列。

Gaussian 和 Informed 使用相同网络、训练流程与采样器，分别训练权重；实验检验改变 **prior 的均值** 是否改善生成结果。

## 2. 输入、网络输出与最终输出

令 `B` 为 batch 中的场景数，`P = Σ N_s` 为行人总数。

| 对象 | 形状 | 含义 |
|---|---|---|
| 观测位置 | `[P,10,2]` | 第 0–9 帧；条件只使用这部分 |
| 图 | `edge_index: [2,E]`、`edge_attr: [E,5]` | 最后观测时刻的邻居与相对运动 |
| 当前 flow 状态 `x_tau` | `[P,20,2]` | 标准化速度序列在生成过程中的状态 |
| flow time `tau` | `[B]` | 人工生成时间，同一场景的行人共享 |
| 网络直接输出 `u_theta` | `[P,20,2]` | flow 状态对 `tau` 的变化率 |
| 完整生成器输出 | `[M,P,20,2]` | `M` 次抽样得到的未来位置；当前 `M=20` |

网络直接输出 **flow vector field**；物理速度和位置由采样与反标准化得到。

## 3. 共同模型结构

```text
观测位置 → 相对位移(2) + 最后绝对位置(2) + 观测速度(2)
         → Linear(6→64) → 两个 TemporalBlock → 历史 context [P,64]

最后观测位置与速度 → 固定局部图与 5 维边特征

x_tau → Linear(2→64)
      + 历史 context + 未来帧 embedding + flow-time embedding
      → [TemporalBlock → SpatialBlock] × 3
      → MLP head → u_theta [P,20,2]
```

- **TemporalBlock**：沿时间轴做膨胀 1D 卷积，使用残差与 LayerNorm，学习运动的时间关系。
- **SpatialBlock**：共享 MLP 计算邻居消息，按接收行人取均值，再做残差更新，学习行人交互。
- **图**：基于第 9 帧；半径 3 m，最多 12 个入邻居，无半径内邻居时连接最近的人。边特征为相对位置 2 维、距离 1 维、相对速度 2 维。图与边特征在生成期间固定，不跨场景连接。
- **Time bundling**：每次网络调用共同处理全部 20 个未来帧。生成仍需多次 flow 求解，并非只调用一次网络。

## 4. 两个 prior 的区别

**Gaussian 的起点均值固定为零；Informed 的起点均值根据当前行人的历史速度计算。** 后面的网络结构相同，但分别训练，得到两套权重。

Gaussian：
$$
\pi(x_0, x_1 \mid c) = p_0(x_0)\,p_1(x_1 \mid c), \qquad p_0 = \mathcal{N}(0, \sigma^2 I).
$$
Informed prior:
$$
\pi(x_0, x_1 \mid c) = p_0(x_0 \mid c)\,p_1(x_1 \mid c)
$$
所有 flow 变量都在标准化速度空间，使用训练集统计量 `mu_v`、`s_v`。
$$
X_0^G=\sigma\epsilon,\qquad \epsilon\sim\mathcal N(0,I)
$$

$$
m_{i,k}=\gamma^k\bar v_i,\qquad
X_{0,i,k}^I=\frac{m_{i,k}-\mu_v}{s_v}+\sigma\epsilon_{i,k}
$$

`bar v_i` 是最近三个观测速度的平均值，`k=1,...,20`，`gamma` 从训练集估计；两者 `sigma=1`。Gaussian 的标准化零均值对应训练集的平均物理速度。

| Prior | 动机 | 主要局限 |
|---|---|---|
| Gaussian | 简单、通用，提供比较基准 | 初始序列可能离目标较远 |
| Informed | 利用短期运动持续性，可能提供更接近目标的起点 | 急停、转弯时均值可能不合适；全局衰减只是近似 |

两者的网络都接收历史。Informed 本身也是高斯分布，区别是均值依赖历史。它不保证误差更低，也没有自动减小噪声或求解成本。

## 5. 训练：学习“如何变化”

真实未来速度由相邻位置差除以 `DT=0.5 s` 得到；首个目标连接第 9 帧和第 10 帧。将其标准化为 `X_1`，抽取 prior `X_0` 和场景 flow time `tau`：

$$
X_\tau=(1-\tau)X_0+\tau X_1,
\qquad
\frac{dX_\tau}{d\tau}=X_1-X_0.
$$

因此训练网络预测 `X_1-X_0`。MSE 先在行人的帧与坐标上平均，再在场景内平均，最后对场景平均，避免人多的场景仅因人数获得更大权重。

未来参与训练监督与插值构造；条件编码和图不读取未来。训练不需要求解 flow ODE。两个 prior 使用相同初始化种子、数据顺序和训练预算；以验证 loss 选择各自最佳权重。

## 6. 生成：从起点积分到未来

保持历史条件固定，抽取 `X_0`，用 30 步 Euler 从 `tau=0` 积分到 `tau=1`：

$$
X_{\tau+\Delta\tau}
=X_\tau+\Delta\tau\,u_\theta(X_\tau,\tau,C),
\qquad \Delta\tau=1/30.
$$

反标准化得到物理速度，再恢复位置：

$$
\hat p_i^{9+k}=p_i^9+0.5\sum_{j=1}^k\hat v_i^{9+j}.
$$

换随机数得到 20 组共同场景未来；每次并行处理最多 4 组以控制显存，各组图相互隔离。**flow time 与物理时间不同**。
