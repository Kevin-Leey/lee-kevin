# RGD-Driver TVT 投稿前论文呈现审查

审查日期：2026-08-16

## 1. 审查范围

本报告只审查论文页面中已经呈现的内容，包括标题、摘要、贡献、相关工作、方法与公式、实验组织、图表文字对应、结果对论点的支撑、Discussion、Conclusion、引用和 IEEEtran 排版。

本报告不评价代码与论文是否对应，不评价当前仓库能否复现实验，也不追查日志、结果文件或数据来源。所有判断以最新的 `paper/main.tex`、`paper/main_zh.md` 和收敛后的 `paper/main.pdf` 为准。Fig. 3 的协议核对另外参考 Hu 等人原论文的 Fig. 15 及其实验设置。

## 2. 总体判断

论文已形成清晰的研究主线：闭环 LLM 推理存在 query state 与 release state 之间的时间失配；RGD 将查询准入和释放授权分成两个独立决策；R-VoD 在离线阶段度量释放状态中的纠正机会；实验从资源选择、时延作用、释放验证、组件机制、推理器替换和场景迁移等角度支撑该框架。

本轮修改后，G. Discussion 不再逐项重复实验数字与图表结论，而是集中讨论 RGD 的系统价值、适用边界与未来方向。该处理与 TVT 论文常见写法更一致，也减少了实验章节与 Discussion 之间的内容重复。

当前仍有一项较高优先级的纸面风险：摘要、贡献和 Conclusion 声称 RGD 在减少慢路径调用时保持任务结果，但 Fig. 4 只直接展示请求量及其统计比较，没有在同一结果单元中给出配对任务端点。该问题不影响资源削减结论，但会影响“资源降低且结果保持”这一中心主张的纸面闭环。

## 3. 标题、摘要与贡献

### 3.1 标题

当前标题为：

> RGD-Driver: Release-Aware Gated Deliberation for Large Language Models in Closed-Loop Autonomous Driving

标题同时给出方法名、核心机制、模型对象和应用场景，定位清楚，符合 TVT 方法论文的标题风格。

### 3.2 摘要

摘要保持单段结构，依次说明 release-time mismatch、现有系统缺口、RGD、R-VoD 和闭环实验结论。摘要没有详细阈值或实验数字，也未引入 VLM 研究主线。

“under the paired protocol”指各方法在相同随机种子、交通初始条件、控制器、动作接口和评价规则下运行，并对同一种子对应结果进行成对比较。该短语在统计意义上成立，但对首次阅读者略显抽象。若后续继续精修，可考虑改为“under matched-seed evaluation”或“under a common matched-seed protocol”，以避免被理解为通信协议。

### 3.3 三项贡献

贡献顺序为 RGD、R-VoD、实验验证，逻辑正确。第三项保留 highway-env 与 MetaDrive 场景，不再列举具体实验种类，并使用单数“the corrective intervention”，与 Fig. 8 中保留唯一纠正干预一致。

三项贡献的职责区分明确：第一项提出在线双阶段权限机制；第二项提出独立的离线诊断量；第三项概括闭环证据。当前长度适中，没有重新展开实验流程。

## 4. Introduction 与 Related Work

Introduction 由三个主要段落和贡献列表组成。第一段建立问题并给出延迟车道变换示例；第二段区分选择性计算、异步控制与运行时保障所回答的问题；第三段提出 RGD、R-VoD 及证据路径。问题、缺口、方法与验证顺序完整。

Related Work 的三个子节均为两段。每个子节先综合相邻研究，再回到“延迟提案如何获得执行权限”的区别。引用组最多四篇，首次引用编号从 [1] 到 [31] 连续推进，没有明显回跳或大跨度堆叠。

## 5. 第三章方法与公式

第三章顺序为 Query and Release Process、Query Admission、Release Validation、Offline Measure of Corrective Opportunity。RGD 在线机制在前，R-VoD 离线诊断在后，与贡献和实验顺序一致。

式（1）至式（8）编号连续，主要关系成立：

- 式（1）定义运行时恢复代价，并将动作相对代价与场景风险映射到统一尺度。
- 式（2）将预测时延换算为剩余机动窗口。
- 式（3a）至式（3d）分别定义时延可行性、动作支持、恢复代价优势与场景需求。
- 式（4）使用合取形式给出查询准入，资源项与四项准入判据缺一不可。
- 式（5）在释放状态重新比较慢提案与同期快速动作。
- 式（6）明确拒绝时的快速路径回退，并将最终安全投影置于动作选择之后。
- 式（7）在共同快速延续策略下仅改变首个有效动作。
- 式（8）因候选集合包含快速动作而按构造非负。

“recoverable”使用“meets the R-VoD margin”，与 $\mathrm{R\text{-}VoD}(s)\geq\epsilon_{\mathrm R}$ 一致。在线代价裕度 $\delta$ 与离线效用裕度 $\epsilon_{\mathrm R}$ 虽同为 0.02，但量纲与用途已经区分。中文稿的式（3a）至式（3d）编码正确。

## 6. Fig. 3 与 Hu 原论文协议核对

### 6.1 实验设置

| 核对项 | Hu 原论文 | 当前论文 | 结论 |
|---|---|---|---|
| 仿真平台 | highway-env | highway-env | 一致 |
| 成功判据 | 连续 30 帧无碰撞 | 连续 30 帧无碰撞 | 一致 |
| 设置 1 | 4 lanes, density 2 | 4L/2.0 | 一致 |
| 设置 2 | 4 lanes, density 2.5 | 4L/2.5 | 一致 |
| 设置 3 | 5 lanes, density 3 | 5L/3.0 | 一致 |
| 每种设置 | 10 个不同种子 | 10 个不同种子 | 一致 |

### 6.2 基线数值

| 方法 | 4L/2.0 | 4L/2.5 | 5L/3.0 |
|---|---:|---:|---:|
| Hu et al. (GPT-4) | 68.6% | 56.2% | 33.1% |
| GRAD | 63.2% | 49.8% | 11.2% |
| DeepSeek-R1 | 67.3% | 54.7% | 30.5% |

Fig. 3 中上述九个数值、方法名称、设置顺序与成功率单位均与 Hu 原论文 Fig. 15 一致。RGD 的 100%/100%/80% 为本文新增结果，没有覆盖或改写来源论文数据。

正文报告的提升量也正确：31.4、43.8 和 46.9 个百分点。对 5L/3.0 的解释已回到 RGD 机制，即更复杂交通使查询状态动作更容易在释放前失去支持，合取式准入与当前状态验证分别处理请求选择和释放授权。

## 7. 实验章节结构审查

### 7.1 当前结构

实验章节目前包括：

1. Experimental Setup。
2. Performance Context Under Published Settings。
3. Delay-Controlled Allocation and Component Evidence。
4. Reasoner-Interface Evaluation。
5. Evaluation Across Lane and Density Conditions。
6. Cross-Scenario Evaluation。
7. Discussion。

其中 C. Delay-Controlled Allocation and Component Evidence 下设五个同级小节：Paired Common-Protocol Allocation、Zero Added Delay Reference、Allocation With a 1.7-s Added Delay、Response Validation and Component Effects、Full-Gate and Projection Analysis。

### 7.2 层级问题

当前内容本身没有逻辑错误，但小节层级并不完全对称。Zero Added Delay Reference 是完整的查询侧分配实验；Allocation With a 1.7-s Added Delay 主要承担受控时延设置的过渡说明，真正的 1.7-s 结果分散在随后两个同级小节中。因此，标题“3) Allocation With a 1.7-s Added Delay”与“4) Response Validation...”在逻辑上更接近父子关系，而不是两个平行实验类别。

此外，零时延实验和 1.7-s 实验的差别不只是时延取值。前者主要回答查询门控如何控制慢路径工作量，后者主要回答延迟响应如何在释放时获得或失去执行权限。若只按“零时延/1.7 s”划分，会弱化 RGD 的双阶段方法结构。

### 7.3 建议结构（尚未执行）

更合适的组织方式是按研究问题划分为两个主要证据板块：

1. **Query-Side Allocation and Delay Sensitivity**：包含配对资源分配、零时延参考和固定门控时延扫描，对应 Fig. 4 与 Fig. 5。
2. **Release-Side Validation Under Added Delay**：先交代 1.7-s 设置，再包含响应验证、自然请求流消融、完整门控与投影分析，对应 Figs. 6–8。

该结构比单纯按时延值划分更强，因为它直接对应 RGD 的 query admission 与 release validation 两项核心机制，也可以删除当前只有过渡作用的独立“Allocation With a 1.7-s Added Delay”小节。按照用户要求，本轮未实施该结构调整。

## 8. 逐图审查

### 8.1 Fig. 2

八个子图的 highway、merge、roundabout 与 intersection 标签和上下两排仿真器对应正确。该图界定评估范围，不承担性能比较。

### 8.2 Fig. 3

协议、数值、设置顺序与提升量均正确。正文已经解释 RGD 在 5L/3.0 中的最大优势来源，而不是仅复述柱状图高度。

### 8.3 Fig. 4

RGD、Always Slow 与 Random 分别为 0.37、3.87 和 2.37 requests/episode。90.5% 与 84.5% 降幅、$-3.50/-2.00$ 配对差、置信区间及 $d_z=-2.49/-1.49$ 均与正文一致。

正文将 0.37 requests/episode 换算为 30 回合中的 11 次请求，并以式（4）的合取条件解释请求为何显著减少，形成“统计结果、实际工作量、机制来源”三层分析。

剩余风险是 Fig. 4 未直接展示主配对协议的任务端点。它充分证明资源削减，但“同时保持配对任务结果”的结论仍依赖正文声明。

### 8.4 Fig. 5

零附加时延下 RGD 为 161 次请求，Always Slow 为 180 次，请求降幅 10.6%；平均单帧时间由 37.4 ms 降至 35.5 ms。时延扫描标签 161/162/143/109 与正文一致。

正文正确区分两个区间：0 与 0.7 s 下请求量接近，说明短时延尚未明显压缩准入窗口；1.7 与 2.7 s 下请求量下降，说明预测释放时刻开始接近可行机动窗口。该解释直接对应 $L_t$ 的作用。

距离置信区间重叠只能支持“观测距离分布接近”，不能被解释为正式等效检验。当前正文使用“preserving the observed distance profile”，表述边界可接受。

### 8.5 Fig. 6

计数链完整自洽：

> 143 valid returns -> 138 reached release -> 28 state-valid -> 9 command-distinct -> 6 margin-passed -> 2 post-projection distinct

Panel (b) 的 22 + 4 + 2 = 28 与状态有效总数一致。正文明确指出 143 个有效返回中有 141 个不改变最终快速动作，并将其解释为释放门控把大量模型输出压缩为少量执行变化。这是全文最直接体现 release validation 必要性的实验。

### 8.6 Fig. 7

请求数 99/117/107/120、额外释放状态 38/36/100，以及纠正状态率差 $-0.18/+0.36/+2.54$ pp 均与正文一致。

正文正确处理 w/o $H$ 的正向结果：移除 $H$ 后纠正状态率增加，不代表 $H$ 富集纠正状态，而是说明移除 $H$ 同时扩大请求量和纠正状态暴露。在线恢复代价用于工作量控制，R-VoD 用于独立释放状态诊断，两者职责已区分。

### 8.7 Fig. 8

Panel (a) 的完整 RGD 轨迹为 70 -> 49 -> 40 -> 8 -> 0。移除 $H$、$N$ 与 $H/N$ 后分别保留 1、8、40 个候选。Panel (b) 的 issued/released/timeout 分别为 1/1/0、8/5/3、40/26/14，数值闭合。

Panel (c) 的三项效用优势为 +0.0546、-0.0325 与 +0.0047。释放验证保留唯一超过 0.02 纠正裕度的干预，并过滤一项有害与一项可忽略的替代动作。正文建立了准入控制服务暴露、投影消除可执行冗余、释放验证授予纠正权限的完整机制链。

## 9. 逐表审查

### 9.1 Table I

RGD 的 distance、speed、successful runs 与 runtime 为表内最优；PADriver--Normal 的 Saf. 与 Kep. 为最优。加粗正确。正文量化了相对 PADriver--Normal 的 +8 m、+1.13 km/h、+4 次成功及 1.4 s 到 0.038 s 的运行时间变化。

### 9.2 Table II

三种 Hu 设置的距离、速度、请求数与单帧时间和正文一致。1.0/1.0/0.8 requests/episode 与 Fig. 3 共同说明成功率优势并非依赖持续调用慢推理器。

### 9.3 Table III

Qwen3-8B 与 Qwen3.5-4B 的 138 个有效响应并列最高；GPT-5.6-sol 的 603.39 m 为最佳平均距离；七个推理器的 29/30 完成次数并列最优。加粗正确。

正文没有把响应数最多等同于闭环进度最好，而是指出 GPT-5.6-sol 以最少有效响应取得最佳距离。该结果支持所评估推理器集合中的描述性稳定，不应扩展为统计等效证明。

### 9.4 Table IV

5L/2.0 的 629 m 和 30/30 为全表最优，加粗正确。正文以横向替代动作和释放时间隙解释该设置的优势，并说明高密度下门控会随间隙收缩限制慢路径权限。

### 9.5 Table V

场景 A、B、C 已在 caption 中完整定义。场景级最优值加粗正确：RGD 在 Scene A 的 safety 100.0% 最优，ASR-RL 在 Scene B 的 safety 与 speed 最优，RGD 在 Scene C 的 speed 6.9 m/s 最优。

正文聚焦 RGD 真正占优的 Scene A safety 与 Scene C speed，并分别从当前状态重映射、合取式准入、释放比较和快速路径连续控制解释其价值，没有把非优势指标设为主叙事。

## 10. Discussion 与 Conclusion

### 10.1 Discussion

G. Discussion 现为两个功能明确的段落。第一段说明 RGD 的价值：将计算准入与执行权限分离，在保持快速闭环连续性的同时，只允许仍然有效的慢提案介入；共享动作接口、恢复代价比较和 R-VoD 分别承担统一权限契约与独立诊断职责。该段不再重复各图表数字或逐项总结实验。

第二段集中讨论适用边界和未来工作。当前边界包括离散高层命令、结构化推理器输出与受控仿真时延；未来方向包括连续轨迹提案、随机服务时延、时延感知重规划、硬件在环与实车评估，以及交通分布变化下的自适应裕度标定。限制与展望保持简洁，没有在正文前部扩大论文责任。

### 10.2 Conclusion

Conclusion 保持两段。第一段回收问题和方法，第二段回收实验意义与“expiring proposal”这一记忆点。结论没有引入正文未出现的新机制，也没有重新展开限制讨论。

英文与 `paper/main_zh.md` 已同步更新 Discussion 的价值、限制和未来工作表述。

## 11. 排版与 IEEEtran 审查

收敛 PDF 共 11 页，使用 IEEEtran `letterpaper,journal` 模式。指定图表页码保持不变：

| 页码 | 图表 |
|---:|---|
| 3 | Fig. 1 位于页顶 |
| 5 | Figs. 2-4 |
| 6 | Fig. 5, Tables I-II |
| 7 | Figs. 6-7 |
| 8 | Fig. 8, Tables III-IV |
| 9 | Table V 位于页顶 |
| 10-11 | Discussion、Conclusion 与 References |

未发现图表遮挡、正文重叠、图片裁切、表格越界、未解析引用或 overfull box。Discussion 压缩后，参考文献平衡触发点由 `\IEEEtriggeratref{24}` 调整为 `\IEEEtriggeratref{28}`，第 11 页参考文献分布在双栏顶部，避免整栏空置。页面下部留白由参考文献总量决定，不是浮动体排版异常。

## 12. 审稿风险与处理优先级

### 高优先级

1. 主配对实验未在 Fig. 4 同一结果单元中展示任务端点。若不增加紧凑端点结果，投稿回复中应准备明确说明逐种子端点及其配对统计口径。

### 中优先级

1. C 节五个同级小节的抽象层级不完全一致。建议后续按 Query-Side Allocation 与 Release-Side Validation 两个机制板块重组，但本轮未修改。
2. 摘要中的“paired protocol”对非统计读者略显抽象，可考虑改为“matched-seed evaluation”。

### 低优先级

1. 第 11 页参考文献数量较少，双栏顶部以下存在自然留白。当前已完成双栏平衡，不影响内容顺序或 IEEEtran 合规性。

## 13. 最终结论

论文最强的实验记忆点仍然清楚：Fig. 4 证明慢路径资源选择性，Fig. 6 给出 143 -> 2 的释放授权链，Fig. 8 证明释放验证保留唯一具有实质纠正价值的干预。Fig. 3 与 Tables I-II 提供已发表设置下的性能背景，Tables III-V 扩展到推理器、交通条件与复合危险场景。

本轮 Discussion 压缩是合理修改。它把实验结果的详细解释保留在各结果小节，把 Discussion 恢复为方法价值、适用边界和未来方向的综合讨论。除主配对任务端点的纸面闭环和 C 节层级组织外，稿件在方法表述、图表对应、中英文一致性和指定版式方面已达到较高完成度。
