# 大模型自动驾驶文献定位与 RGD 差异矩阵

更新日期：2026-07-17

## 1. 文档用途

本文档只服务于当前 TVT 稿件的定位、Related Work 组织和术语一致性检查。它不是系统综述，也不以文献数量替代逐篇核验。最终引文集合以 `paper/references.bib` 和 `paper/main.tex` 的实际引用为准。

当前论文只讲一个问题：

> 当前状态值得调用慢推理，不等于慢推理返回时仍有可执行的纠正机会。

论文据此把 **post-latency recoverability** 作为车辆侧 test-time computation allocation 的核心变量。研究对象不是更强的 driving backbone，也不是让大模型直接接管车辆。

## 2. 最终方法口径

### 2.1 R-VoD：理论对象

Recoverable Value of Deliberation（R-VoD）定义在慢请求的 release state。只有当该状态仍存在一个经过共享安全映射后、相对 matched fast continuation 达到目标增益的合法动作时，慢推理才保留纠正机会。

R-VoD 是有限时域的 oracle object。当前实验没有观测完整 oracle membership，也没有训练 oracle estimator。因此不得把 operational proxy 写成 oracle calibration、oracle prediction 或普适 value-of-computation 定理。

### 2.2 RGD：可实现分配器

Recoverability-Gated Deliberation（RGD）在请求发出前计算一个工程化 proxy：

- latency survival；
- legal non-fast alternatives；
- recovery-cost headroom；
- 在通过 opportunity floor 后用于排序的 present-need term。

RGD 只决定是否购买慢请求。它不生成新的 driving backbone，不替换 complete fast policy，也不获得最终 safety authority。慢请求 pending 期间 fast policy 持续执行；返回后，proposal 还必须通过 release-state legality 和共享 safety map。

### 2.3 证据合同

论文把以下对象分开记录：query decision、return survival、release divergence、release legality、safety rewrite 和 final action。该 query-to-actuation contract 防止把“发出请求”“模型返回不同动作”和“动作最终获得执行权”混为一谈。

## 3. 文献簇与真实差异

| 文献簇 | 典型研究问题 | 常见决策主体 | 与 RGD 的关系 | 当前稿件的区别 |
| --- | --- | --- | --- | --- |
| LLM/VLM driving agents | 如何利用语言或视觉语言模型提升理解、规划与控制 | 大模型 driver 或核心 planner | 提供可被调用的高容量慢分支 | RGD 研究何时购买该分支，不声称改进其生成能力 |
| World models and generative planners | 如何表示未来、生成场景或扩展候选轨迹 | world model、diffusion/generative planner | 增加 proposal space 或 rollout quality | RGD 不扩大候选空间，只判断延迟后是否仍可能纠正 |
| Dual-process and adaptive reasoning | 何时根据反馈、不确定性或复杂度启用慢思考 | fast-slow controller 或 learned router | 与 RGD 最接近 | 现有触发量主要描述 current salience；RGD 增加 release-state opportunity 约束 |
| Async and resource scheduling | 请求能否在队列、时限或预算内返回和复用 | scheduler、queue manager | 处理计算可服务性和时间成本 | RGD 进一步问返回动作是否还保留车辆控制权 |
| Viability and runtime assurance | 哪些动作或控制器可安全执行 | safety filter、viability set、runtime assurance | 提供最终执行边界 | RGD 不替换安全层；它在购买前估计延迟后是否可能存在合法纠正槽位 |

## 4. 与重点参考工作的关系

### 4.1 大模型与视觉语言驾驶

DiLu、DriveGPT4、LMDrive、VLM-Driver 以及 TVT 中的 driving-knowledge 和 vision-language navigation 工作，核心贡献在于增强场景理解、知识利用、决策生成或闭环驾驶能力。它们说明高容量模型能够产生有价值的 proposal，但没有直接回答一个已购买的 proposal 在返回时是否仍能改变车辆运动。

因此，本稿不能把“使用 Qwen3-8B”写成主要创新。该模型是共享 slow executor；主要创新是与 executor 解耦的 vehicle-side allocator。

### 4.2 World model 与生成式规划

DriveDreamer、GenAD、DREWM、DiffusionDrive 和 C-TRAIL 等工作扩展了未来建模、候选轨迹生成、常识检索或 trust-guided planning。它们主要改善“慢分支能产生什么”。R-VoD/RGD 研究不同问题：“当结果返回时，还有没有一个相对 complete fast continuation 的合法纠正机会”。

两类贡献可以组合，但不能互相替代。更强的 proposal generator 只能提高机会存在条件下命中纠正集合的概率，不能恢复已经因延迟消失的机会。

### 4.3 双过程与异步推理

CogniDrive、LeapAD、LeapVAD、FASIONAD、AdaDrive 和 ThinkDrive 说明反馈、uncertainty、complexity、learned reward、异步队列与目标重锚定可以改善 fast-slow cooperation。AsyncDriver 则强调慢语言指导的异步复用。

本稿与这些工作的差异必须写成变量层面的区别：

- current difficulty、uncertainty 或 hazard 说明现在是否值得注意；
- queue occupancy、deadline 或 cache 说明请求是否可服务；
- goal reanchoring 说明返回结果如何修复；
- post-latency recoverability 说明在发出请求前，延迟后的合法纠正机会是否可能仍存在。

### 4.4 Metareasoning、viability 与安全保证

经典 metareasoning 和 deliberation scheduling 工作提供 computation cost、deadline 和 expected value 的理论背景；anytime computation 强调价值取决于中断时间。Viability theory、runtime assurance、Simplex 类架构和 RSS 则提供动态环境中的合法执行边界。

R-VoD 是面向闭环车辆动作的必要机会对象，不是新的通用 metareasoning 理论或 safety theorem。当前 Proposition 只在明确的 finite-horizon、nested-feasible-set 和 non-increasing-advantage 条件下成立。

## 5. 当前论文的三项贡献

1. **Allocation object**：定义 release-state 的 post-latency corrective opportunity，并给出空集拒绝条件和条件性 latency-erosion 命题。
2. **Operational allocator**：给出由 latency survival、legal alternatives、recovery headroom 和 need ranking 组成的 RGD gate，同时保持 backbone 与 safety authority 不变。
3. **Stage-resolved evidence**：用 common Fast-only trajectories、controlled latency erosion、closed-loop endpoint 和 query-lifecycle audit 检验 query placement、机会保持与最终执行权。

不得把以下内容拆成额外“创新模块”：hidden `SLOWER` bridge、highway pass rule、trace schema、collision audit、图表生成脚本或匿名制品打包工具。它们属于实现修复、协议完整性或证据基础设施。

## 6. 核心证据与结论强度

| 问题 | 证据 | 允许的结论 |
| --- | --- | --- |
| RGD 是否选择更可持续的 query state | 1.7 s matched Fast-only trajectories：54/111 对 23/100；paired difference 0.256 [0.136, 0.375] | RGD-selected states 更常保留 operational joint opportunity |
| 延迟是否侵蚀机会 | 固定 trajectory 和 route rule：0.615、0.486、0.383 对应 0.7、1.7、2.7 s | 在当前嵌套条件和 highway protocol 下，机会随延迟下降 |
| 完整系统是否保持运行 | RGD-only closed-loop sweep：0--1.7 s 为 29/30，2.7 s 为 27/30 | 给出 bounded latency-stress profile，不识别 allocator-latency interaction |
| 主 endpoint 是否优于 baselines | main seeds：RGD 28/30；TTC-risk 27/30；其余 26/30 | 方向有利但统计未解决，不得宣称 completion superiority |
| 4/5/6 车道能否执行 | density 2/3、540 episodes、零附加延迟 | 验证设置执行和 traffic-stress sensitivity，不支持统一跨设置优势 |

## 7. Related Work 写作顺序

1. 先说明大模型 driving、world model 和 generative planner 提高 proposal capability。
2. 再说明 dual-process、risk/uncertainty routing 和 asynchronous systems 管理 activation、cadence、queue 或 return repair。
3. 接着引入 metareasoning、anytime computation、viability 和 runtime assurance，建立“计算期间状态继续演化”的理论背景。
4. 最后提出缺口：现有变量没有在购买前直接表示 delayed release state 是否仍保留 matched-fast-relative corrective opportunity。

该顺序使研究故事从“能生成什么”转到“何时仍有权执行”，而不是把论文写成模块清单。

## 8. 禁止恢复的旧口径

以下术语或主张与最终论文不一致，不应重新写入正文、摘要、图注或配置说明：

- `ASRO` 或 `ASRO-conditioned counterfactual deliberation`；
- 把 RGD 写成 oracle estimator、safety certificate 或 universal value-of-computation；
- `compute-matched superiority`、`completion superiority` 或跨 simulator generalization；
- 把 query count、slow-fast disagreement 或 safety override 单独当作有效干预；
- 把 density 3.0 的高碰撞率隐去或解释成 RGD 特有缺陷；
- 把 1.7 s 写成真实硬件推理延迟。

## 9. 单一事实源

- 运行默认值：`config.yaml`；
- 正式协议与 submission contract：`formal_protocol.yaml`；
- 可复现实验结果：`results/tvt_revision_round5/`；
- 最终论文：`paper/main.tex` 与 `paper/main.pdf`；
- 论文事实审计：`tools/audit_tvt_manuscript.py`；
- 协议一致性验证：`tests/test_protocol_contract_alignment.py`。

任何数据、seed、阈值、延迟或术语发生变化时，必须同步更新上述事实源及四份核心 TVT 文档，不能只改论文表格或只改配置。
