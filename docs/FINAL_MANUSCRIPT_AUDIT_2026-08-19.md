# RGD-Driver 最终稿件系统审查报告（严格 TVT 审稿视角）

- 本轮最后修改时间：2026-08-25 14:56（Asia/Shanghai）
- 审查对象：`paper/main.tex` 与其当前编译稿 `paper/main.pdf`
- 稿件形态：IEEEtran journal 双栏，10 页，5 个一级章节，8 幅图，5 张表
- 审查定位：模拟 IEEE Transactions on Vehicular Technology（TVT）资深审稿人的投稿前审查草案，不代表期刊编辑决定
- 校准状态：`NOT_CALIBRATED`
- 标准绑定：`criteria_binding_unavailable`。未提供 TVT 当前官方审稿表或稿件编号，因此 venue 判断仅按 TVT 的自动驾驶、车载智能、实时决策与控制读者语境及 IEEE Transactions 通行标准作审稿模拟，不声称官方合规结论

## 1. 审查边界

本报告只审查论文当前呈现出来的研究问题、贡献定位、方法定义、论证逻辑、实验叙事、统计报告、图表、讨论、结论和版面可读性。稿内数值均按作者报告值接受，不检查其真实性。

本轮明确不做以下工作：

- 不读取或评价代码实现、配置、测试、日志和运行环境；
- 不读取或重算原始数据、结果文件、随机种子或统计产物；
- 不验证图表数值是否由代码或数据真实生成；
- 不核验参考文献真伪、DOI、外部论文内容或引用是否完全支持正文；
- 不查找仓外证据，不联网检索未公开稿件内容；
- 不评价作者身份、机构、研究动机或潜在行为。

旧版报告中的 32 条参考文献真实性核验不属于本轮授权范围，因此不再纳入本轮结论。下文提到“缺少支撑”时，仅指稿件自身没有把主张、定义、结果和结论闭合，不表示数据不存在或结果不真实。

## 2. 评审组织与限制

本轮按 `academic-paper-reviewer` 的角色分离框架，从五种视角审查：期刊契合度、方法与统计、领域贡献、系统与部署边界、最强反方论证。前三种视角由分离的审查席完成，后两种由主审单独复核后统一裁决。

这些审查席共享同一模型族和工作区。角色分离有助于覆盖不同问题，但不构成统计独立性，也不意味着错误相互独立。最终意见由主审逐项回到 `main.tex`、编译 PDF、图表和交叉引用复核后形成。

| 审查席 | 主要关注 | 独立信号 |
|---|---|---|
| Journal-Fit | TVT scope、原创性、读者价值、稿件成熟度 | `Major Revision` |
| 方法与统计 | 数学闭合、实验单位、estimand、对照、统计解释 | 现稿不宜接收；实质重构后重审 |
| 领域审稿 | LLM driving、异步控制、runtime assurance、术语边界 | 多项 Major；核心 claim 尚未闭环 |
| 系统视角 | 信任边界、时序、部署条件、可推广性 | `Major Revision` |
| Devil's Advocate | 核心贡献的最强替代解释 | 最强反驳成立，裁定为 Major 而非 Critical |

严重度定义：`Critical` 指单一缺陷本身足以使核心贡献不可成立或无法通过修订挽救；`Major` 指实质削弱核心主张并阻断接收，但仍可通过补充论文内呈现、重分析或收窄主张修复；`Minor` 指不改变核心结论的清晰度、术语、图表或版式问题。本报告没有认定不可修复的 singleton-Critical 缺陷。

## 3. 模拟编辑结论

### 建议：Major Revision

稿件与 TVT 范围高度相关。其最有价值的思想不是“LLM 能驾驶”，而是把慢推理器定义为具有到期语义的 proposal generator，并把 query admission 与 release-time actuation authority 分离。问题定义直观，fast fallback、release-state remapping、pre/post-projection 区分和离线 R-VoD 的边界也比许多 LLM-driving 稿件更清楚。

但当前稿件更充分地证明了“系统大量拒绝慢分支并回退 fast path”，尚未闭合“慢推理经 RGD 后为闭环任务提供了什么不可由 fast controller、手工 cost selector 或简单 release monitor 替代的增量价值”。30-seed 主配对实验只展示请求量，未展示同一 cohort 的任务端 paired outcome；143 次请求只有 2 次改变最终可执行动作；直接 matched-proposal 审计只有 3 个投影后不同的干预。与此同时，方法的关键代价构成、权重、阈值、校准边界、时序参数和 release 端映射没有在论文内完整公开。

因此，当前版本不能进入 Minor Revision。核心思想仍可修复，不足以判定为不可挽救的 Reject；但若作者既不补齐主要闭环比较，也不系统收窄摘要、贡献和结论，则下一轮应转为 Reject（insufficient demonstrated contribution / premature），而不是继续用跨论文 benchmark 数值补偿。

### 不能接收的直接原因

1. 核心“减少调用且保持任务结果”联合主张未在同一主 paired cohort 中呈现完整结果。
2. 现有结果无法排除 fast controller、fallback 或手工 recovery-cost evaluator 主导全部任务表现的替代解释。
3. RGD 的实际判定边界无法仅凭论文复原，方法形式化与实验实例之间存在断裂。
4. release validation 的一般性结论主要来自极少 actionable interventions，摘要和结论没有充分暴露这一规模边界。
5. 跨论文、跨模型和跨场景结果被提升为 superiority、adaptation 或 robustness 结论，但稿件呈现不足以区分描述性结果与受控机制证据。

## 4. 判据级判断

| 维度 | 判断 | 主要锚点 | 决策含义 |
|---|---|---|---|
| TVT scope / readership | `MEETS` | `main.tex:57-108`, `194-222` | LLM latency、闭环自动驾驶、runtime authority 与车载智能直接相关 |
| Originality | `PARTLY_MEETS` | `main.tex:76-91`, `152-222` | 两阶段 authority contract 有清晰价值，但与 AsyncDriver、Simplex、concurrent planning、delay-aware control 的最邻近差异尚未操作化 |
| Methodological Rigor | `DOES_NOT_MEET` | `main.tex:275-440`, `466-480` | 核心量、权重、阈值、校准和 release 映射未充分定义，空集分支也未数学闭合 |
| Evidence Sufficiency（仅指稿内呈现） | `DOES_NOT_MEET` | `main.tex:48`, `122-125`, `478-480`, `585-737`, `890-893` | resource reduction 有呈现；paired task preservation、slow-path 增量价值和 release filter 的一般性未闭合 |
| Argument Coherence | `PARTLY_MEETS` | `main.tex:93-126`, `353-440`, `662-737`, `880-898` | problem→method 很强；method→result→broad claim 较弱，pre/post-projection 语义在结论中发生滑移 |
| Statistical Reporting | `PARTLY_MEETS` | `main.tex:469-480`, `527-530`, `597-603`, `635-648` | 以 seed 为配对单位、bootstrap、exact test 与 Holm 方向合理；主要终点、test mapping、区间与 preservation 推断不完整 |
| Literature Integration | `PARTLY_MEETS` | `main.tex:127-222` | 分类覆盖合理，但 closest-work delta 主要靠断言，缺功能级对照和限定性 novelty claim |
| Writing / Presentation | `PARTLY_MEETS` | 全文；PDF pp.1-10 | 英文和 IEEE 结构成熟；核心图偏密，图注与表注不完全自包含，结果组织弱化贡献主线 |
| Significance / Impact | `PARTLY_MEETS` | `main.tex:597-737`, `806-862` | workload governance 有现实价值；LLM slow path 的非平凡车辆技术收益仍不清楚 |

## 5. 已确认的真实优点

1. **问题定义准确。** `main.tex:57-91` 用“query-state semantic correctness 不等于 release-state control authority”建立了真实的闭环时间错配，例子、系统后果和研究问题相互对应。
2. **权限分层清楚。** `main.tex:93-108`、`223-253` 将 query admission 与 actuation authority 分开，明确请求成功不会自动继承为未来命令权。
3. **fallback 与状态重映射写得较完整。** `main.tex:254-274`、`353-400` 说明 observed release state、concurrent fast command、mapper failure 和 fast fallback，时间语义总体可信。
4. **没有隐藏 post-projection no-op。** `main.tex:662-676` 明确报告 6 个 cost-margin-passed commands 中 4 个经 `\Phi` 后与 Fast 等价，仅 2 个真正改变执行动作。这是全文最诚实、最有诊断价值的结果之一。
5. **R-VoD 与在线 gate 有意识地隔离。** `main.tex:401-440` 明确 R-VoD 不进入 online gate，避免后验 rollout label 泄漏到运行时决策。
6. **实验骨架较完整。** 稿件包含 paired allocation、controlled delay、component ablation、matched-proposal replay、reasoner substitution、lane/density stress 和 MetaDrive 场景，说明作者理解系统论文不能只给一个 endpoint benchmark。
7. **统计方向有可取之处。** `main.tex:469-476` 以 simulator seed 为配对实验单位，报告 seed bootstrap、exact paired tests 和 Holm correction，优于把 frame 或 event 当作独立样本。
8. **总体版式成熟。** 当前 PDF 为 10 页 IEEE 双栏，公式、图表、引用和交叉引用均能解析；未发现正文或公式大面积越界、重叠或编译失败。

## 6. 决策性问题（必须处理）

### R1. 主 paired cohort 没有闭合“资源减少且任务结果保持”的联合主张

- **严重度：** Major；方法席认为达到 Critical，主审裁定为可修复的 Major。
- **位置：** Abstract `main.tex:47-48`；Contribution 3 `122-125`；Experimental Setup `469-480`；Fig. 4 `524-531`；主结果 `585-615`；Conclusion `890-893`。
- **观察：** 稿件称 30-seed paired cohort 提供“primary resource and task comparison”，但 Fig. 4 和对应正文只报告 requests/episode、paired request difference、`d_z` 与 `p_H`。同一 cohort 中没有 RGD、Fast-only、Always Slow、Random、Uncertainty、Risk 的 completion、collision/safety、distance/progress 或其他预先定义任务端点的 paired difference。
- **为何重要：** 减少请求是 conjunctive gate 的直接预期行为；若没有同一 cohort 的任务端比较，无法判断节省是否以任务性能为代价，也无法说明 RGD 相对请求更少的 Risk 或零请求的 Fast-only/Uncertainty 处于更优资源—性能位置。
- **最小充分修订：** 在主文加入同一 30-seed cohort 的 resource–task 表或图，明确主要任务终点、方向、paired effect 与 95% CI。若使用“preserving/non-degradation”，必须给预设 non-inferiority/equivalence margin 或至少给可判断实际差异的区间。若不呈现该结果，则删除 Abstract、Contribution 3、Discussion 和 Conclusion 中所有 outcome-preservation 联合主张，只保留 request reduction。

### R2. 没有识别 slow reasoner 相对 fast path 和手工 selector 的增量价值

- **严重度：** Major。
- **位置：** `main.tex:275-345`, `353-400`, `662-676`, `722-737`, `806-825`；Figs. 6、8；Table III。
- **观察：** gate 已枚举 `\mathcal A(s)`、计算 recovery cost 并筛出最优替代；143 次请求只有 2 次改变最终可执行动作，七个 reasoner 均为 29/30 completion。稿件没有同时呈现 Fast-only、直接执行 `argmin c(a)` 的 no-LLM selector、shuffled/no-op proposal 或完整 no-release-validation 系统的闭环任务结果。
- **最强替代解释：** 当前结果同样可以由“fast controller 和 downstream safety projection 几乎承担全部闭环表现，RGD 主要是 workload filter”解释。reasoner substitution 的相同 endpoint 也可能是 fallback 抹平模型差异，而非 interface robustness。
- **最小充分修订：** 给出 Fast-only、cost-selector/no-LLM、no-release-validation 与 Full RGD 的同协议 paired task effects，并报告 request→response→authorized→post-projection-distinct→beneficial intervention 完整漏斗。若无法排除该解释，论文应定位为 delayed-proposal governance / workload arbitration，而不是 LLM-enhanced closed-loop driving performance。

### R3. 核心方法无法仅凭论文完整复原

- **严重度：** Major。
- **位置：** `main.tex:275-345`, `353-440`, `466-480`；Eqs. (1)–(8)。
- **观察：** 论文未给出 `d_t(a)` 中 safety/comfort/efficiency 的分项公式与固定权重，未操作化 `h_t`、`n_t^{pre}`、`T_t^{crit}`、`\widehat\tau_t`、`\rho`、`T_A`、`\kappa_c`、`\lambda_{L/A/H/N}`，也未完整定义 `\psi`、`\mathcal M`、`\Phi`、fast policy、target-speed domain、request budget、cooldown、timeout/TTL 与 calibration objective。除 `\delta=0.02`、`H=20`、`\gamma=0.99` 和 `\epsilon_R=0.02` 外，决定 gate operating point 的关键值均不在稿中。
- **为何重要：** RGD 的贡献就是判定 contract。关键自由度只写“calibrated and frozen”使读者无法判断选择性来自一般机制还是特定配置，也无法审查 calibration/evaluation 是否分离。
- **最小充分修订：** 增加单一来源的算法框和 parameter/interface table，列出定义、公式、单位、取值、边界、选择阶段、校准目标、cohort 大小与冻结时点；明确 mapper、shield、fast controller 和 gate 的输入输出与失效语义。

### R4. query 和 release 的形式定义尚未数学闭合

- **严重度：** Major。
- **位置：** `main.tex:287-345`, `353-400`。
- **观察 1：** 当 `\mathcal D_t=\varnothing` 时，正文说不求 Eq. (3c) 且不请求；但 `u_* = \min_{m\in\mathcal F_t}u_m`、Eq. (3b) 与 Eq. (3c) 仍在公式层面对空集合未定义。文字短路不等于数学上的 piecewise definition。
- **观察 2：** release 端的 `F_r,A_r,H_r,c_r` 只说从 `s_r` 和 `a_r^f` 重算，没有显式给出 `t→r` 映射、`\mathcal D_r`、family support 和 returned proposal membership 的确切关系。
- **观察 3：** 文中称 `\delta\ge0` 下为 strict recovery-cost advantage；一般定义在 `\delta=0` 时允许相等。当前实验 `\delta=0.02` 确为严格，但一般表述不闭合。
- **最小充分修订：** 将 `A_t,H_t` 写成以 `F_t=1` 为条件的 piecewise quantities，或在 admission rule 前给出正式短路语义；明确 release-domain 对应定义；令一般条件为 `\delta>0`，或把 strict 改为 non-inferior / margin-qualified advantage。

### R5. pre-projection distinctness 与 actuation-level 主张发生语义冲突

- **严重度：** Major。
- **位置：** `main.tex:99-100`, `113-117`, `231-239`, `369-400`, `662-676`, `883-886`；Eq. (5)–(6)，Fig. 6。
- **观察：** `g_r` 在 `\Phi` 前比较 canonical commands；方法明确承认 authorized slow command 可能在 safety projection 后与 Fast 变成相同 executable action，实验中实际发生 4/6 次。但 Conclusion 写成 delayed proposal “reaches actuation only when it is feasible, distinct from the current fast action”。
- **为何重要：** distinctness 是 release gate 的三项核心判据之一。当前规则只保证 pre-projection command difference，不保证 actuation-level intervention。
- **最小充分修订：** 二选一：在 release rule 中直接比较 `\Phi(s_r,\widetilde a_r^{sl})` 与 `\Phi(s_r,a_r^f)`；或全面把主张限定为 pre-projection canonical-command distinctness，并在 Abstract、Contribution、Discussion、Conclusion 中严格区分 `mapped`、`authorized`、`selected`、`executed` 和 `effective intervention`。

### R6. originality boundary 没有相对最邻近系统操作化

- **严重度：** Major。
- **位置：** `main.tex:76-91`, `152-183`, `194-222`。
- **观察：** Related Work 将 selective computation、asynchronous execution、Simplex/runtime assurance、event-triggered scheduling、delay-aware control 与 viability 分段介绍，随后直接断言“None of them”决定 returned proposal 是否替换 concurrent fast command。但稿件没有说明 AsyncDriver、NetRoller、concurrent planning、SimplexDrive 等是否已经包含 plan invalidation、current-state remapping、acceptance test 或 supervisory switching。
- **为何重要：** RGD 的算法本体是 conjunctive admission、state remapping、comparative threshold 与 fallback 的工程组合。若不证明缺失的 operator/interface，TVT 审稿人可能将其判断为合理但增量有限的系统整合。
- **最小充分修订：** 增加 closest-work 功能对照，至少比较 query trigger、pending-time fast control、release-state remapping、candidate-vs-current-fast comparative test、expiry、post-projection handling、fallback 与 offline diagnostic；把“None”改为有明确限定条件的可审查表述。

### R7. R-VoD 的术语、estimand 与第二主贡献地位均需收紧

- **严重度：** Major。
- **位置：** `main.tex:118-121`, `194-216`, `231-239`, `401-440`, `678-693`；Eqs. (7)–(8)，Fig. 7。
- **观察 1：** 稿件借用 viability/recoverability 语境，但这里的“recoverable”实际指：在固定 fast continuation 和指定 simulator reward 下，至少一个 first effective action 比 fast first action 的 `J_H` 高 0.02。它不是 reachability-to-safe-set、viability kernel 或形式恢复保证。
- **观察 2：** `r_k` 的归一化、collision penalty 与 simulator reward 是否重复计入安全、candidate generation、随机环境的 matched randomness、每个 action 的 rollout 数和 max-selection optimism 均未在论文内说明。
- **观察 3：** 结果只报告删除 L/A/H 后相对 Full RGD 的 corrective-state-rate 差值，没有 Full RGD 的绝对 R-VoD 分布、corrective prevalence、与 online `c_t` 的一致/冲突关系或 `H,\gamma,\epsilon_R` 敏感性。删除 H 后 rate 增加 2.54 pp，也不直接支持“query gate concentrates corrective opportunities”。
- **最小充分修订：** 将术语限定为 `corrective-opportunity state` 与 `surrogate action cost`，明确不构成形式 safety/recoverability；完整定义 rollout estimand 与随机过程；报告绝对分布、门内/门外对照、与 online cost 的增量信息及参数敏感性。若不能展示独立 insight，应把 R-VoD 从主贡献降为辅助机制审计量。

### R8. 统计 estimand、cohort ledger 与“保持性能”的推断不完整

- **严重度：** Major。
- **位置：** `main.tex:333-334`, `469-480`, `527-530`, `597-603`, `629-660`, `695-724`, `807-862`。
- **观察：** 稿件未列主/次终点及每项分析的 estimand；未说明 calibration cohort 的大小、seed 和与 evaluation 的隔离；多组 20/30-seed cohort 的复用关系不清；MetaDrive 样本量未给；McNemar、sign-flip 与具体 endpoint 的映射及 Holm family 未列。
- **统计错误边界：** `main.tex:641-648` 以 marginal 95% CI overlap 和均值接近描述 distance preservation。区间重叠不是无差异或等效性证据，也不能区分低精度与实际等效。
- **报告缺口：** 除 request contrasts 外，Tables I–V 的多数 distance、completion、safety、speed 只有点估计；reasoner、traffic、scenario 结论没有与其强度相称的不确定性。
- **最小充分修订：** 增加 cohort/analysis ledger，列分析目的、unit、seed/episode 数、cohort 复用、主要终点、effect、CI、test、multiplicity family 与 tuning/test 边界；对 preservation 使用 paired difference 与 non-inferiority/equivalence 逻辑，不再用 CI overlap 作结论。

### R9. 跨论文结果没有被充分隔离为 contextual comparison

- **严重度：** Major。
- **位置：** `main.tex:446-515`, `782-805`, `845-862`；Tables I、II、V，Fig. 3。
- **观察：** Table I、Fig. 3 和 Table V 把 RGD 与 cited values 并列，但没有在表内逐行标明 locally rerun、reproduced 或 source-reported，也没有明确共享哪些 simulator version、seed、episode count、policy/training、hardware 和 failure accounting。`main.tex:469-472` 的“Within each comparison, methods share...”可能让读者误以为 published baselines 也与 RGD 共享实验 realization。
- **过强表述：** “best”“exceeding by X percentage points”和随后 release-aware 机制解释，把跨实现点估计提升为 head-to-head superiority 与因果归因。
- **最小充分修订：** 在 row、legend 和 caption 标注 provenance；把 source-reported 与 locally rerun 明确分组。若不是同平台、同实现、同 seed 的受控比较，只作为量级背景，不作显著优越或机制性解释。

### R10. release、reasoner、traffic 与 scenario 结论超出稿内可见设计

- **严重度：** Major。
- **位置：** `main.tex:620-737`, `739-862`；Figs. 7–8，Tables III–V。
- **release audit：** Fig. 8(c) 最终只有 3 个 post-projection distinct interventions，其中 1 retained、2 filtered。这是三个清楚的案例审计，不足以无规模限定地称“complete mechanism chain”或一般性 filtering ability。
- **reasoner：** Table III 只给 Valid Resp.、Distance、Completion；没有 issued denominator、authorized、post-projection-distinct 或 beneficial counts。相同 29/30 既可能说明 interface robustness，也可能说明 slow models 几乎未影响 actuation。
- **lane/density：** Table IV 只有 RGD 单臂点估计和固定参数，不能单独证明 gate adaptation，也无法把结果归因于 `A_t/H_t`。
- **MetaDrive：** Table V 中 RGD 在 Scene B 的 safety 和 speed 均低于 ASR-RL；Scene C 仅快 0.1 m/s，却低 13.3 percentage points safety；Scene A 以 2.3 m/s 速度损失换 0.8-point safety 提升。没有定义的 Pareto、utility 或 safety constraint，不足以称“favorable balance”。
- **最小充分修订：** 将 3-case audit 明确标为 illustrative；按 reasoner 报完整 authority funnel；将 Table IV 降为 descriptive stress map，或补同协议对照；逐场景正面报告 Table V 的优势、劣势和 dominated/trade-off 状态，删除无判据的 “favorable balance”“adapts” 和直接机制归因。

### R11. trust、safety 与 timing contract 不完整

- **严重度：** Major。
- **位置：** `main.tex:194-217`, `241-274`, `297-307`, `338-400`, `466-480`, `629-653`；Fig. 1。
- **观察：** 稿件同时使用 feasible、admissible、safety shield、recovery 和 authority，但真正 safety projection 在 RGD 之后；`F/A/H/c` 不是形式 safety certificate。论文没有明确 fast controller 是否 trusted、shield 保证什么、mapper/cost evaluator 失效后如何处理、single-pending request 假设与 queue 语义是什么。
- **时序缺口：** predicted latency、base LLM service latency、artificial added delay、per-frame runtime、release frame、timeout、expiry 与 episode termination 没有统一 taxonomy。文中说 expired proposal 被拒绝，但 formal rule 主要只明确 episode termination 时丢弃。
- **最小充分修订：** 增加 system trust/assumption 段与 timing table，分别定义 query timestamp、service latency、added delay、predicted latency、release timestamp、TTL/timeout、gate overhead、control-loop runtime；明确 RGD 是 supervisory arbitration，而不是 safety guarantee。

### R12. Discussion 与 Limitations 没有消化论文自己暴露的边界

- **严重度：** Major。
- **位置：** `main.tex:863-877`。
- **观察：** Discussion 基本重述方法；Limitations 只列 discrete commands、structured outputs、controlled simulator delays 与未来 HIL/real vehicle。未讨论 rare effective interventions、fast path/safety projection dominance、heuristic feasibility 不等于 safety、single-pending request、threshold calibration、R-VoD 对 fast policy/reward/horizon 的依赖、跨论文比较边界和 MetaDrive safety trade-off。
- **最小充分修订：** 至少增加两个实质段落：一段说明 guarantee、trust、timing、queue 和 failure assumptions；一段说明 empirical interpretation、fast-path-dominated alternative、actionable-event scarcity、cross-study comparability 和 external-validity 边界。未来工作不能替代对当前未证明内容的明确承认。

## 7. 最强反方论证及裁决

### 反方论证

RGD 的性能并不一定来自“有效调用慢推理”，而可能来自一个强 fast controller、一个手工 recovery-cost evaluator 和一个总是可回退的 safety stack。conjunctive gate 必然减少调用；143 次请求只有 2 次改变最终执行动作；不同 reasoner 仍给出几乎相同 endpoint；直接 release audit 只有 3 个有效候选。因此，稿件现有结果可以支持“系统安全地忽略绝大多数 LLM 输出”，却未证明“LLM deliberation 经该接口为闭环驾驶带来稳定增量价值”。R-VoD 又是在已知 simulator reward 和 fixed fast continuation 下对 first-action alternatives 取最大值，其结果不能自动转化为在线 LLM value 或形式 recoverability。

### 裁决

该反方论证由 `main.tex:662-676`、`722-737`、Table III 以及主 paired outcome 缺口共同验证，不能忽略。它阻止 Accept/Minor Revision。

主审没有将其定为 Critical，理由是：论文的 release-time authority 问题和 interface 设计本身仍成立；作者可以通过同 cohort 的 fast-only/no-LLM/no-release-validation 对照证明增量价值，也可以诚实地将贡献收窄为 workload governance 与 delayed-proposal arbitration。核心思想尚非不可修复。

## 8. 逐图审查

| 图 | 当前作用 | 审查结论 | 必要修改 |
|---|---|---|---|
| Fig. 1 | query、slow reasoner、release validation、mapper、shield、R-VoD 架构 | 主流程直观；但商业模型 logo 未解释，slow reasoner 与 Table III 模型集合不完全对应；trust boundary 与 timing/expiry 未画出 | 用抽象 reasoner block 或在 caption 解释 logo；标 trusted/untrusted、pending/expiry、pre/post-projection 边界 |
| Fig. 2 | highway-env 与 MetaDrive 场景 | 图像清楚，但 caption 未说明上/下两行分别对应哪个 simulator，场景只作示意 | caption 明确 row-to-simulator mapping；说明是 representative scenes 而非结果 |
| Fig. 3 | Hu 等设置下 success-rate context | 数值可读；但 published values 与 RGD provenance、sample size、uncertainty 和非配对性质不醒目 | 图例/caption 标 `source-reported` 与 `this work`，给 n/CI 或明确 descriptive only |
| Fig. 4 | 主 paired allocation | request reduction 清楚；没有 task outcome panel，不能支撑联合主张；右侧元素接近物理页边 | 补同 cohort task outcomes；修复右侧溢出并统一单位为 requests/episode、方向定义 |
| Fig. 5 | zero-delay workload/runtime 与 delay sweep | 三面板清楚；panel (c) 只有 RGD distance，CI overlap 被过度解释为 preservation | 给 paired contrast/非劣逻辑或改为描述；说明 runtime 是否含 LLM、gate、fast path |
| Fig. 6 | release funnel 与 cost advantage | 是全文最有价值的机制图；诚实显示 143→2；但小字在打印尺度偏小，stage denominator 需更自包含 | 放大字体；caption 定义 state-valid、mapped distinct、authorized 与 executable distinct |
| Fig. 7 | L/A/H one-at-a-time ablation | 显示 workload 与 corrective-rate trade-off；缺 N，且 rate difference 无 absolute baseline | 补 Full RGD absolute rate/denominator；解释为何无 N；避免把 rate change 等同 enrichment |
| Fig. 8 | frozen bank serial retention、lifecycle、3-case matched audit | 对三个案例描述有价值；不同 panel 涉及的 cohort/denominator 容易被误认为同一总体，结论过宽 | caption 明确 panel cohort 与 denominator；将 panel (c) 标为 illustrative 3-case audit；放大旋转标签 |

PDF 第 7 页同时堆叠 Figs. 5–7，核心标签在最终印刷尺度偏小；Fig. 4(b) 的最右元素明显越过常规正文右边界并接近物理页边。方法机制高度依赖这些图，不能把可读性当作普通美化项。

## 9. 逐表审查

| 表 | 当前作用 | 审查结论 | 必要修改 |
|---|---|---|---|
| Table I | PADriver configuration 下的 performance context | Saf./Kep. 只在正文解释；baseline provenance 不清；“best/relative increase”容易被理解为 matched head-to-head | 表注定义全部缩写、n、aggregation、higher-is-better 与 reported/rerun 来源；非配对则删除 inferential superiority |
| Table II | Hu 设置下 RGD operating metrics | 只给 RGD，无法与 Fig. 3 baselines 形成资源—性能联合比较；runtime 构成不清 | 标明 unsuccessful=0 的 speed estimand、n 与 runtime scope；必要时加入 matched resource metrics |
| Table III | 多 reasoner interface | `Valid Resp.` 未定义 denominator；模型标识和 `Fable-5` 未说明；缺 authority funnel 与 uncertainty | 定义 model/version、issued denominator、valid criterion、authorized/executed counts、paired effect/CI |
| Table IV | lane/density stress | 单臂点估计可作 boundary map，不能证明 gate adapts；缺 request/release metrics 和 uncertainty | 改为 descriptive robustness；若归因 gate，加入各 predicate pass rate、requests、executed interventions 与对照 |
| Table V | MetaDrive compound hazards | Scene B 被 ASR-RL 双指标支配，Scene C 安全损失远大于速度增益；“favorable balance”不成立 | 给样本量/uncertainty/provenance；逐场景陈述 trade-off，或定义复合 utility/Pareto 约束 |

## 10. 次要但应统一处理的问题

1. **标题定位。** “RGD-Driver”容易暗示完整 driver；实际核心更接近 reasoner-agnostic release-time authority / gating interface。标题应避免把 wrapper 扩张成完整驾驶系统。
2. **摘要缺关键规模。** Abstract 没有 90.5%、1.7 s、143→2、3-case audit 或主要 paired task endpoint。严格 TVT 摘要应量化主要结果，同时暴露稀疏干预边界。
3. **生命周期术语。** `valid response`、`release event`、`state-valid`、`actionable`、`authorized`、`selected`、`executed`、`effective intervention` 混用。需要统一 glossary 和 denominator。
4. **`A_t` 命名。** Eq. (3b) 更接近 maneuver-family coverage / relative-cost concentration，不是某个 slow proposal 的 scene support；建议改名或精确定义。
5. **“independent”需限定对象。** R-VoD 只独立于 specific reasoner output 和 online gate，仍依赖 fast policy、action domain、projection、reward、horizon 与 simulator；不能简称 independent mechanism audit。
6. **机制因果语气。** `main.tex:511-513`, `609-615`, `645-648`, `734-737`, `819-825`, `855-862` 使用 `explains`, `demonstrates`, `establishes`, `adapts`，而对应设计多为描述性或稀疏案例。应改为 `is consistent with`, `illustrates`, `under the evaluated stream`。
7. **图表自包含性。** caption/footnote 应定义 denominator、单位、aggregation、uncertainty、paired status、结果来源和方向，不能依赖读者在正文中搜寻 Saf./Kep./Valid Resp. 等定义。
8. **统计报告细节。** 应给 bootstrap 重复数、双侧规则、raw/adjusted p、Holm family、test-to-endpoint mapping；`p_H=0.00025` 不应成为唯一 inferential 信息。
9. **速度单位不统一。** Table I 使用 km/h，Tables II/V 使用 m/s。可保留来源单位，但 caption 和讨论应避免无提示比较。
10. **PDF 技术预检。** 编译日志只有 underfull box/vbox，无 overfull 或 unresolved reference；但 `pdffonts` 显示一个未嵌入的 Helvetica base font，PDF metadata 的 Title/Author/Subject 为空。提交前应通过 PDF eXpress/期刊预检确认字体与 metadata。

## 11. 结果章节的结构问题

当前实验章先给 PADriver/Hu 的 published-settings performance context，再给真正对应贡献的 allocation、delay validation、ablation 和 matched-proposal audit。这个顺序把稿件包装成 benchmark paper，却让核心 authority mechanism 到第 6 页后才获得直接检验。

建议按贡献逻辑重组，而不是按“先展示高分”组织：

1. Experimental protocol、cohort ledger、primary estimands 与 parameter table；
2. Query admission：resource–task paired main comparison；
3. Release validation：full-system ablation 与 authority funnel；
4. R-VoD：绝对分布、增量 insight 与 sensitivity；
5. Reasoner / traffic / scenario boundary tests；
6. Published results 作为明确标记的 contextual comparison；
7. 独立 Discussion 与 Limitations。

此建议不是为了形式偏好，而是为了恢复 claim→test→result→limitation 的可追踪顺序。

## 12. 作者必须回答的问题

1. 30-seed primary paired cohort 中，各 allocator 的 completion、collision/safety、distance/progress 分别是什么？“preserving”对应哪个预设 estimand 与 margin？
2. 如果 143 次 slow request 中只有 2 次改变 executable action，RGD 相对 Fast-only 的直接任务增量是什么？论文如何排除 fast controller 主导全部 endpoint 的解释？
3. 为什么需要 LLM，而不是直接执行 recovery-cost 最优替代或一个 no-LLM selector？slow reasoner 提供了 gate 本身不能计算的什么信息？
4. 相比 AsyncDriver、NetRoller、SimplexDrive、concurrent planning 和 delay-aware control，RGD 在 response-arrival-to-actuation 之间新增的不可替代 operator 是什么？
5. `recoverable`、`feasible` 与 `safety` 是形式动力学保证，还是 simulator-specific surrogate 判据？RGD、fast controller、mapper 与 downstream shield 各自保证什么、不保证什么？
6. distinctness 的正式目标是 canonical command level 还是 post-projection executable-action level？若是后者，为何 gate 授权 4 个最终与 Fast 相同的命令？
7. Table I、Fig. 3、Table V 的每一行哪些是本研究重跑，哪些是原文报告值？它们共享了哪些 seeds、implementation、training、simulator version 与 failure accounting？
8. R-VoD 在 Full RGD 下的绝对分布和 corrective prevalence 是什么？它相对 online recovery cost 提供了什么不可替代的新判断？
9. Table III 中每个 reasoner 的 issued、valid、authorized、post-projection-distinct 和 beneficial intervention 数是多少？相同 endpoint 是 interface robustness 还是 fallback dominance？
10. Table V 中 Scene B 的双指标劣势和 Scene C 的 13.3-point safety 损失为何仍被称为 favorable balance？对应的预设 trade-off criterion 是什么？

## 13. 两条可接受的修订路径

### 路径 A：维持当前核心主张

若要保留“减少调用且保持任务结果”“release validation 保留纠正干预并过滤有害/微小替代”“reasoner-independent”“adapts across conditions”等强主张，至少需要在论文中呈现：

- 同一 primary paired cohort 的 resource–task joint comparison；
- Fast-only、no-LLM cost selector、no-release-validation 与 Full RGD 的闭环对照；
- 完整 parameter/interface/timing/trust table 与算法框；
- R-VoD 的绝对分布、与 online cost 的增量关系及 sensitivity；
- 更充分的 post-projection actionable proposal population 或明确的统计不确定性；
- reasoner 与 stress strata 的完整 authority funnel；
- 跨论文结果来源与可比性分层；
- 重写 Discussion、Limitations、Abstract 与 Conclusion。

### 路径 B：不新增主结果，严格收窄论文

若本轮不增加新的主要比较，论文仍可向“release-aware delayed-proposal governance”收敛，但必须：

- 删除所有 paired task preservation 的联合结论；
- 把 90.5% 限定为相对 Always Slow 的 request reduction，而非整体 superiority；
- 把 Fig. 8(c) 限定为 evaluated three-case audit；
- 将 reasoner、lane/density 和 MetaDrive 结果降为 descriptive compatibility / boundary observations；
- 将 published-settings 部分降为 contextual background；
- 把 R-VoD 降为 policy- and reward-conditioned offline diagnostic；
- 明确 RGD 不构成 safety、viability 或 formal recoverability guarantee；
- 在标题、摘要、贡献、讨论和结论中统一使用较窄定位。

路径 B 的代价是贡献强度会下降，届时 TVT novelty threshold 仍可能不足；但它比维持当前证据边界之外的强主张更可信。

## 14. 修订完成判据

只有同时满足以下条件，稿件才有资格从 Major Revision 降为 Minor Revision：

- Abstract、Contribution 3、主结果和 Conclusion 对同一 resource–task estimand 使用一致措辞；
- 论文内可重建所有 gate 判据、参数、校准边界、时序和 release mapping；
- pre-projection 与 post-projection distinctness 在公式、图、表和结论中完全一致；
- fast-path-dominance 的替代解释被直接比较或在贡献中明确承认；
- R-VoD 的含义不再借用形式 viability/recoverability 的保证色彩；
- 跨论文结果明确标注 provenance，不再承担未经匹配的 superiority/causal claim；
- Tables III–V 的结论强度与其设计、denominator 和 uncertainty 匹配；
- Discussion/Limitations 明确列出当前未证明的内容与部署前提；
- 核心图在 IEEE 最终尺寸可读，Fig. 4 不侵入外边距；
- PDF 预检不再报告需要处理的字体、metadata 或布局问题。

## 15. 最终审稿意见

RGD-Driver 具备一个值得 TVT 读者关注的系统问题和有潜力的接口设计。稿件已经超过“只有想法没有系统”的阶段，尤其是 release funnel、post-projection collapse 和 offline/online 边界的呈现具有真实价值。

当前阻碍不在英文润色或一般版式，而在贡献闭环：请求减少并不自动等于有效选择；fallback 稳定并不自动等于 slow reasoner 有价值；三个 matched interventions 不足以支持一般过滤能力；跨论文点估计也不能替代同 cohort 的核心比较。作者必须选择“补足主张所需的论文内比较”或“把主张收窄到现有呈现真正能够承担的范围”。在完成这一选择之前，稿件不宜提交为 TVT 可接收版本。

## 16. 变更记录

| 时间（Asia/Shanghai） | 变更摘要 |
|---|---|
| 2026-08-25 00:39（文件此前修改时间） | 前一工作稿聚焦 paired task outcome 缺口、admission 空集定义与参考文献核验 |
| 2026-08-25 14:56 | 按严格 TVT 多视角审稿重构全文；限定为论文呈现审查；保留并扩展原 C1/C2；新增判据判断、12 项决策性问题、Devil's Advocate 裁决、逐图逐表审查、修订路径与完成判据；移除超出本轮范围的参考文献真实性表 |

## 17. 2026-08-25 IEEE TVT 同标终审与最终裁决

### 17.1 本节的裁决效力

本节是在逐页复核当前 `paper/main.pdf`、逐项核对 `paper/main.tex`，并以同一标准对照 `paper/ref` 中 IEEE TVT 成稿后形成的最终意见。若本节与第 3、4、6、7、8、9、11、13、14、15 节的早期判断冲突，以本节为准。早期报告有意采用了接近理论控制论文、完整复现审计和最强反方论证的混合尺度，其中若干要求超出了本轮“只审查论文呈现，不检查代码、复现性、外部证据对应，允许适度 overclaim”的边界，不能继续作为阻止投稿的依据。

本节采用的判断问题只有一个：在与 `paper/ref` 中已经发表或已接收的 IEEE TVT 自动驾驶系统论文相同的呈现标准下，本稿是否存在必须在投稿前继续修改的根本逻辑错误、公式错误、图文矛盾或明显低于期刊常规水准的写作与版式问题。

### 17.2 实际采用的 TVT 对照集

以下 9 篇论文均由 PDF 首页的 `IEEE Transactions on Vehicular Technology` 页眉、正式卷期或 accepted-for-publication 声明确认，不以文件名推断期刊归属。

| 对照论文 | 稿件形态 | 页数 | 主要对照维度 |
|---|---:|---:|---|
| *A Generalized ChatGPT-Based Collaborative Multi-Objective Decision-Making Framework for Robust Vehicle Platoon Collision Avoidance* | TVT 74(5), 2025 | 14 | LLM 系统故事、模块链、仿真与实验组织 |
| *Behavioral Uncertainty-Aware Attention Allocation via VLMs for Interactive Autonomous Driving* | TVT accepted author version, 2026 | 14 | 选择性计算、方法形式化、消融与统计呈现 |
| *C-TRAIL: A Commonsense World Framework for Trajectory Planning in Autonomous Driving* | TVT accepted author version, 2026 | 14 | 问题立意、信任机制、闭环规划框架 |
| *DriveSOTIF: Advancing SOTIF Through Multimodal Large Language Models* | TVT 75(3), 2026 | 14 | 摘要、贡献点、系统模块和多组实验 |
| *Enhancement of Large Language Models Driving Knowledge for Practical Autonomous Driving Decision Making* | TVT accepted author version, 2026 | 17 | LLM 驾驶知识增强、方法到任务结果的叙事 |
| *Integrating Vision and Language Foundation Models for Enhanced Navigation and Decision-Making in Connected Autonomous Vehicles* | TVT 74(10), 2025 | 17 | 多模块系统、公开设置对比与结论写法 |
| *LLM-Based Misbehavior Detection Architecture for Enhanced Traffic Safety in Connected Autonomous Vehicles* | TVT 74(8), 2025 | 13 | 系统架构、任务定义和实验呈现 |
| *Towards Interactive and Learnable Cooperative Driving Automation: A Large Language Model-Driven Decision-Making Framework* | TVT 74(8), 2025 | 12 | LLM 闭环决策框架、记忆模块和案例分析 |
| *VLM-Driver: Human-Like Autonomous Driving Decision-Making via Vision Language Model* | TVT 75(5), 2026 | 16 | 标题摘要、贡献结构、跨场景实验和图表密度 |

对照集同时包含正式出版稿和已接收作者版，覆盖了本稿当前投稿阶段最相关的两种形态。对照论文为 12 至 17 页，本稿为 10 页；页数差异用于判断信息密度，不被直接解释为质量差异。

### 17.3 同一标准下的总体比较

| 维度 | 本稿当前表现 | 与 TVT 对照集的相对位置 | 最终判断 |
|---|---|---|---|
| 研究问题 | 以 query state 与 release state 的时序错配建立 response-arrival-to-actuation authority 问题 | 比部分多模块系统稿更集中，与 C-TRAIL、Behavioral Uncertainty-Aware 的单一核心缺口叙事相当 | `MEETS` |
| 摘要 | 单段完成问题、现有缺口、RGD、R-VoD、平台验证与结论 | 结构完整，信息密度不低于对照集，且少于模块罗列和实现过程描述 | `MEETS` |
| 引言 | 三段建立问题、相关方向边界和本文方案，贡献顺序为 RGD、R-VoD、实验验证 | 缺口形成速度和 closest-work differentiation 高于若干对照稿 | `MEETS` |
| Related Work | 按 language-conditioned driving、dual process/selective computation、runtime safety/delay/viability 组织，每节回扣 release authority | 分类和回扣逻辑达到 TVT 常规水准，非简单文献枚举 | `MEETS` |
| 方法 | Fig. 1 与 Eqs. (1)–(8)依次定义 admission、release、fallback 和 offline diagnostic | 形式化密度高于主要依靠 pipeline 描述的系统型 TVT 稿，与 C-TRAIL 类框架论文处于可比区间 | `MEETS` |
| 实验 | 公开设置语境、配对分配、delay sweep、release funnel、自然流消融、冻结 bank、reasoner、traffic、MetaDrive 构成机制链 | 图表数量不是优势来源，但每组实验职责清楚，解释深度高于若干只报告 endpoint 的对照稿 | `MEETS` |
| 图表与正文 | 8 幅图、5 张表均有对应正文分析，关键数值和阶段链可交叉复核 | 没有实质性图文不符；第 7 至 9 页较密但仍在 TVT 成稿常见范围 | `MEETS` |
| 英文与科研叙事 | 术语稳定，问题、机制、结果和结论围绕 release-aware authority 展开 | 句子总体更紧凑，未见系统性中式英语、AI 化抽象堆叠或实验日志式写法 | `MEETS` |
| IEEE 呈现 | IEEEtran journal 双栏，10 页，浮动体顺序正确，参考文献 32 条 | 无可见越界、遮挡、断裂或大面积空白；整体已具备 Transactions 稿件形态 | `MEETS` |

### 17.4 标题、摘要、引言和贡献点终审

本稿标题直接给出方法名、核心机制、模型对象和闭环自动驾驶场景，没有把标题写成宽泛的“LLM-based autonomous driving framework”。这种限定比对照集中部分只给系统名和宏观能力的标题更准确。

摘要当前形成完整的发布逻辑：首先指出 LLM inference latency 导致 query state 与 release state 不一致；随后区分现有 dual-process/asynchronous methods 与本文缺失的 release authority；再分别交代 query admission、release validation 和 R-VoD；最后以 highway-env 和 MetaDrive 闭环实验收束。摘要没有分段，没有实现日志，也没有在贡献建立前讨论限制。按 TVT 对照稿的实际写法，其完整性和紧凑度已经足够，不需要为了加入更多数字而破坏当前模板化摘要结构。

Introduction 前两页已经把研究问题提升为“semantic correctness at query time does not establish control authority at release time”，并明确 selective computation、asynchronous execution 与 runtime assurance 分别只能解决何时调用、如何持续控制和动作是否可接受的问题。本文的新增问题位于 response arrival 与 actuation 之间。这一差异可被 TVT 读者在首轮阅读中直接识别，科研故事不是在实验部分才被补充出来。

三个贡献点依次对应：

1. RGD 的 online two-stage authority contract，对应 Section III-A 至 III-C、Fig. 1 和 Eqs. (1)–(6)；
2. R-VoD 的 offline matched-rollout diagnostic，对应 Section III-D、Eqs. (7)–(8)、Fig. 7(c) 和 Fig. 8(c)；
3. 闭环实验验证，对应 Section IV 的 allocation、delay、release、reasoner、traffic 和 scenario 结果。

该顺序、粒度和证据映射不低于 Behavioral Uncertainty-Aware、C-TRAIL、DriveSOTIF 和 VLM-Driver 的贡献列表。贡献点没有重新列举具体实验种类，也没有把系统实现过程当作贡献。

### 17.5 第三章、Fig. 1 与公式终审

第三章的组织顺序为 `Query and Release Process → Query Admission → Release Validation → Offline Measure of Corrective Opportunity`，先 RGD、后 R-VoD，与摘要和贡献顺序一致。

Fig. 1 当前在线顺序与正文一致：state 同时进入始终工作的 fast controller 和 query-admission gate；slow response 必须先通过 release validation；被授权的 slow command 才进入 shared action mapper；选择结果随后经 safety shield 投影至 vehicle actuation。memory 只向 slow reasoner 提供 recent state 和 retrieved experience，最下方 release snapshot、rollout comparison 和 R-VoD labels 明确属于离线诊断路径。未发现 central authority bypass、mapper bypass 或 online/offline path 混淆。

公式专项复核结果如下：

| 公式 | 作用 | 正确性与闭合性判断 | 后续实验对应 |
|---|---|---|---|
| (1) | 归一化 recovery cost | `d_t(a)`、`h_t` 有值域，clip 保证 `c_t(a)∈[0,1]`；`F_t` 和 `D_t` 处理 action-domain integrity 与可行替代 | cost advantage、release margin、Fig. 6 |
| (2) | latency-aware residual window | 预测延迟通过控制频率离散到 frame，分子与分母量纲一致，负余量由 clip 处理 | Fig. 5 delay sweep、Fig. 7 的 `w/o L` |
| (3a)–(3d) | `L/A/H/N` admission predicates | delay、family support、cost headroom 和 scene demand 分工明确；阈值在 calibration cohort 后冻结 | Fig. 7 和 Fig. 8(a)(b) |
| (4) | query admission | 资源、domain guard 和四个 admission predicates 以 conjunctive gate 合并 | Fig. 4、Fig. 5、Fig. 7、Fig. 8 |
| (5) | release validation | slow proposal 在 `s_r` 重映射，并与同一状态的 concurrent fast command 比较 feasibility、distinctness 和 cost advantage | Fig. 6、Fig. 8(c) |
| (6) | selection、fallback 与 final projection | `g_r=0` 时 fast fallback 明确；`Φ` 位于 branch selection 之后 | Fig. 6 的 post-projection collapse |
| (7) | matched-rollout utility | 固定 horizon、discount、collision penalty 和 common fast continuation，比较对象明确 | Fig. 8(c) 的 utility values |
| (8) | R-VoD | 因 `C(s)` 包含 fast effective action，R-VoD 非负；corrective margin 与 online `δ` 明确隔离 | Fig. 7(c)、Fig. 8(c) |

未发现集合类型冲突、维度错误、未定义主要分支或公式与 Fig. 1 流程相反的问题。`D_t=∅` 时，正文已规定 `F_t=0`、不求值 (3c) 且不发请求；这是明确的 operational short circuit，不是数学未闭合。pre-projection command distinctness 与 post-projection executable-action collapse 也已在 Eq. (6) 后和 Fig. 6 中主动区分，不能再判为方法矛盾。

与 `paper/ref` 中 TVT framework/system 类论文相比，本稿方法形式化深度已经充分。它不需要达到以 Bayesian inference、game theory 或 model predictive control 为核心的理论论文公式数量；继续补充公式、algorithm box 或完整参数表只属于扩展选择，不是当前投稿门槛。

### 17.6 实验章节逐图逐表终审

#### Figures

| 图 | 图文与数字复核 | 对 RGD 主张的作用 | 最终判断 |
|---|---|---|---|
| Fig. 1 | 在线 gate、memory、mapper、shield、actuation 与离线 R-VoD 均有正文对应 | 建立完整系统边界和两阶段 authority contract | 自洽，无需修改 |
| Fig. 2 | 上排 highway-env、下排 MetaDrive，四类场景标签与 caption 一致 | 交代实验覆盖，不承担性能结论 | 自洽，无需修改 |
| Fig. 3 | 三个 traffic settings、四种方法和成功率均与正文对应；31.4、43.8、46.9 percentage-point 差值正确 | 建立 published-settings 性能语境，并突出复杂交通下 RGD 的成功率 | 自洽，无需修改 |
| Fig. 4 | `0.37/3.87/2.37` 对应约 `11/116/71` 次请求；`−3.50/−2.00`、90.5% 和 84.5% 均正确 | 证明 query admission 将 continual invocation 压缩为 sparse but nonzero allocation | 自洽，无需修改 |
| Fig. 5 | `161/180` 对应 10.6% reduction，`161→109` 对应 32.3%；距离约为 599–605 m | 将 `L_t` 与 delay-aware workload control 对应 | 自洽，无需修改 |
| Fig. 6 | `143→138→28→9→6→2` 完全闭合；panel (b) 的 `22+4+2=28`；141/143 保留 fast final action | 是全文最强的 release mechanism evidence，清楚展示 validation、distinctness、margin 与 projection 的逐层作用 | 自洽，无需修改 |
| Fig. 7 | Full RGD 99；`w/o L/A/H` 为 117/107/120；附加 release states 为 38/36/100；rate difference 为 `−0.18/+0.36/+2.54 pp` | 区分 admission predicates 的 workload 与 corrective-state exposure 作用 | 自洽，无需修改 |
| Fig. 8 | retention stages、`issued=released+timeout` 和三次 matched utility 均闭合；`0.721−0.667≈0.0546`，8.2% relative gain 正确 | 连接 frozen-bank admission、service lifecycle、projection 和 release-value selection | 自洽，无需修改 |

#### Tables

| 表 | 图文与数字复核 | 对 RGD 主张的作用 | 最终判断 |
|---|---|---|---|
| Table I | 最佳值粗体正确；相对 PADriver–Normal 的 `+8 m`、`+1.13 km/h`、`+4 runs` 正确 | 在 PADriver experimental configuration 下给出性能语境 | 自洽，无需修改 |
| Table II | 三个设置与 Fig. 3 一一对应；`1.0/1.0/0.8 requests per episode` 和 `<66 ms/frame` 正确 | 将 success rate 与 RGD operating metrics 联系起来 | 自洽，无需修改 |
| Table III | valid responses 的相对范围约 23%，distance span 为 `603.39−599.12=4.27 m`，均为 29/30 completion | 支撑 mapper、release checks 和 fallback 对 reasoner substitution 的接口稳定性 | 自洽，无需修改 |
| Table IV | 低密度和高密度范围、5L/2.0 最优值及粗体正确 | 展示固定 gate 参数下的 lane/density operating conditions | 自洽，无需修改 |
| Table V | Scene A 的 RGD safety 最优，Scene B 的两项最优均属于 ASR-RL，Scene C 的 RGD speed 最优；粗体与正文一致 | 以不同 hazard composition 展示速度与安全性的 operating balance，而非宣称所有场景全面领先 | 自洽，无需修改 |

实验章节已经形成“性能语境 → query allocation → delay sensitivity → release funnel → admission ablation → frozen-bank/matched release → reasoner/traffic/scenario extension”的论证链。相较对照集中若干逐表描述 endpoint 的论文，本稿对机制为何产生结果的解释更充分。Behavioral Uncertainty-Aware 等论文在试验规模、参数敏感性和参数表方面更强，Hu 等和 ChatGPT platoon 稿件具有更宽的模块链或真实车辆实验；这些是论文资源和类型差异，不是 TVT 对所有 simulator/system 论文的统一硬门槛。

### 17.7 版式、浮动体和参考文献呈现终审

当前 PDF 为 10 页 IEEEtran journal 双栏：Fig. 1 在第 3 页顶部，Fig. 2 在第 5 页顶部，Figs. 3–4 和 Tables I–II 位于第 5–6 页，Figs. 5–7 位于第 7 页，Fig. 8 位于第 8 页，Tables III–V 位于第 9 页，第 10 页完成 Discussion、Conclusion 和 32 条 References。图表编号、正文首次引用和最终浮动顺序一致。

第 7 页和第 8 页多面板图信息密度较高，第 9 页三张表集中，但坐标、图例、阶段标签和表内数字在最终 PDF 中仍可辨认，没有遮挡、裁切或越过物理页边。对照 TVT 成稿中同样存在高密度多面板图和单页多表，因此该差异不能升级为格式缺陷。

编译日志没有 overfull box、未解析 citation/reference 或编译失败；现有 underfull 提示未形成可见的大面积空白或不协调断行。32 个正文 citation keys 与 32 个 bibliography entries 一一对应，没有 missing 或 uncited entry。参考文献最终可见格式按 IEEE 文献类型区分：journal 为 `vol./no./pp./year`，conference 为 `in Proc./year/pp.`，arXiv 为 `year/arXiv`。不同类型的年份位置不同是 IEEE 样式本身，不是格式不统一。

### 17.8 对前文 Major 判断的最终再裁决

| 前文意见 | 同标终审裁决 | 是否阻断当前投稿 |
|---|---|---|
| 缺少全部权重、threshold、operator 和 timing table，因而方法不成立 | 这属于完整复现或部署审计要求。本稿已给变量角色、值域、校准冻结边界、关键 margin/horizon 和 failure/fallback 语义 | 否 |
| `D_t=∅` 造成 admission 数学不闭合 | 正文已有 `F_t=0`、不求值 (3c)、不发请求的显式短路规则 | 否 |
| pre-/post-projection distinctness 矛盾 | 两个动作层级是有意分离；Eq. (6)、正文和 Fig. 6 均报告 projection collapse | 否 |
| R-VoD 被错误写成 formal safety/viability guarantee | 当前正文始终把它定义为 offline matched-rollout diagnostic，并明确依赖 common continuation 且不进入 online gate | 否 |
| 必须增加 Fast-only/no-LLM/no-validation 等新实验才能投稿 | 这些实验可增强因果识别和接受概率，但不是与本组已发表 TVT system papers 同标后的普遍硬门槛 | 否 |
| 143 次请求只有 2 次改变 actuation，因此 slow branch 没有价值 | 本稿贡献本身是 authority governance；稀疏 actuation changes 与过滤 stale/redundant proposals 是 Fig. 6 的核心机制结果，不等同于方法失效 | 否 |
| Table V 的 trade-off 不能称为全面优势 | 当前正文没有宣称所有场景和指标全面领先，而是描述 one-configuration operating balance，并逐场景指出各自优势 | 否 |
| 第 7–9 页图表密度过高 | 最终尺寸仍可辨读，且处于参考 TVT 成稿常见范围 | 否 |

因此，前文的 `Major Revision` 是超严格反方审查的结果，不再是本次“是否达到 paper/ref 中部分 TVT 论文水准”的最终答案。Methodological Rigor、Argument Coherence、Writing/Presentation 和 Evidence Sufficiency 在本轮限定的论文呈现标准下均至少达到 `MEETS`；不再要求按第 13、14 节继续扩写或重构论文。

### 17.9 保留但不触发修改的审稿风险

仍有两处严格审稿人可能追问的语义边界，但它们不是数字错误、公式错误或图文矛盾，且不足以要求当前继续修改：

1. Contribution 1 和 Conclusion 中“distinct before reaching actuation”的压缩表述，严格来说对应 Eq. (5) 的 canonical-command distinctness；Eq. (6) 和 Fig. 6 已进一步说明 post-projection 仍可能 collapse。完整方法语义是清楚的，风险只存在于核心句的高度压缩阅读。
2. 摘要、Contribution 3 和 Conclusion 将“over 90% request reduction”与“preserving paired task outcomes”并列。两项证据均在正文呈现，但来自承担不同职责的 cohort/analysis。严格审稿人可能要求进一步说明联合 estimand，当前表述仍属于系统论文可接受的结果概括，而不是稿内事实冲突。

这两项应作为投稿后可能收到的 reviewer questions 预先知悉，不作为本轮继续润色的触发条件。继续改写核心句反而可能重新引入已经消除的术语或排版回归。

### 17.10 最终结论

按与 `paper/ref` 中 9 篇 IEEE TVT 成稿或已接收论文相同的严格呈现标准，本稿已经在以下方面进入可比区间：

- 标题与摘要能够快速建立重要问题、现有缺口、独特方法和主要意义；
- 引言与 Related Work 能够把 selective computation、asynchronous execution、runtime assurance 和本文 release authority 清楚区分；
- 三项贡献按 RGD、R-VoD、实验验证排列，并与方法和实验一一对应；
- Fig. 1、Eqs. (1)–(8) 和第三章形成正确、紧凑且足够深入的 online/offline 方法闭环；
- Figs. 3–8 和 Tables I–V 的数字、阶段、粗体最优值、正文计算与机制解释相互对应；
- Discussion 与 Conclusion 强化 release-time mismatch、two-stage authority 和 sparse corrective intervention 三个记忆点；
- IEEEtran 双栏、图表浮动、10 页信息密度和参考文献最终呈现均达到 Transactions 投稿稿件的常规完成度。

**最终裁决：当前稿件已经基本达到 IEEE Transactions on Vehicular Technology 的投稿呈现水准。未发现必须在投稿前继续修改的根本逻辑错误、公式错误、图文不符或明显低于对照 TVT 成稿的语言与版式缺陷。建议保持当前论文内容，不再基于非阻断性的风格偏好、复现扩展或极限理论标准继续修改。**

该结论表示“达到可送审、可投稿的成熟度”，不表示对期刊接收结果作保证。实际审稿仍可能因审稿人偏好、实验广度或贡献判断产生 revision requests，但这不改变本轮关于论文呈现质量已经达标的结论。
