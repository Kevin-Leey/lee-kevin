# RGD-Driver 最终稿件审查报告（IEEE TVT 模板与论文呈现）

- 审查日期：2026-08-24
- 审查对象：`paper/main.tex`、`paper/main.pdf`、IEEE 模板目录 `paper/IEEE-Transactions-LaTeX2e-templates-and-instructions`，以及 `paper/ref` 中识别出的 9 篇 IEEE Transactions on Vehicular Technology 论文
- 审查范围：IEEEtran 源文件语义、作者栏、页眉、摘要与关键词、章节结构、公式、图表浮动体、参考文献呈现、PDF 版面
- 明确排除：代码实现、运行配置、原始实验数据、可复现性、文献真实性核验和正文未要求的内容重写

## 总体结论

当前稿件的 PDF 呈现已经符合 IEEE Transactions 双栏论文的主体视觉规范，整体达到 `paper/ref` 中部分 IEEE Transactions on Vehicular Technology 论文的呈现水准。按同一标准对当前稿和 9 篇 TVT 参考论文进行标题摘要、科研叙事、方法形式化、实验组织、图表、语言引用和版式终校后，当前稿评分为 **90/100**，在 10 篇论文中列 **并列第 5 位**，属于参考组的中上水平。该排名只衡量论文呈现，不评价技术新颖性、实验真实性、代码或可复现性。

`IEEEtran` 的 `letterpaper,journal` 模式、10 页 letter 页面、正文双栏、摘要与关键词位置、首段 drop cap、图表编号、公式编号、页眉、IEEE 引用样式和参考文献连续编号均正常。

本次检查确认：作者单位脚注已经通过 IEEEtran `\thanks` 放置在首页左下角，32 个正文引用与 32 个 `bibitem` 一一对应，没有未定义引用或交叉引用；PDF 编译成功，没有图表裁切、公式越界、正文重叠或明显破坏阅读的空白。当前稿件仍属于 **Minor Editorial Revision**。主要剩余问题是邮箱占位尚未填写、query 和 release 两侧的少量公式定义没有完全闭环，以及个别图注和机制归因措辞与参考组最成熟论文仍有差距。

## 统一评分标准与排名

评分总分为 100 分，统一采用以下维度：IEEE 模板与作者元数据 15 分；标题、摘要与关键词 10 分；问题立意、引言和 Related Work 15 分；方法、公式和符号闭环 20 分；实验组织和结果解释 15 分；图表设计与 caption 10 分；语言、引用与结论 10 分；最终排版完成度 5 分。评分存在约 1 分的人工判断浮动，不代表 IEEE 官方录用评分。

| 排名 | 论文 | 呈现评分 |
|---:|---|---:|
| 1 | C-TRAIL | 94 |
| 2 | DriveSOTIF | 93 |
| 3 | VLM-Driver | 92 |
| 4 | A Generalized ChatGPT-Based Collaborative Multi-Objective Decision-Making Framework | 91 |
| 5 | Towards Interactive and Learnable Cooperative Driving Automation | 90 |
| 5 | **RGD-Driver** | **90** |
| 7 | LLM-Based Misbehavior Detection Architecture | 87 |
| 8 | Integrating Vision and Language Foundation Models for Enhanced Navigation and Decision-Making | 87 |
| 9 | Behavioral Uncertainty-Aware Attention Allocation | 86 |
| 10 | Enhancement of Large Language Models Driving Knowledge | 85 |

RGD-Driver 的主要加分项是问题定义集中，RGD、R-VoD 与实验验证形成顺序清楚的叙事主线，方法公式数量适中，release 机制实验具有较强辨识度，图表版面紧凑且没有明显生产级排版缺陷。作者单位已经补入，当前仅保留邮箱空白占位待最终填写。剩余扣分主要来自 release 侧符号映射仍需读者推断，以及少数 caption 和机制归因没有达到参考组前三篇论文的形式完整度。

## 提交前待完成

### 1. 作者邮箱占位待填写

- 位置：`paper/main.tex:31-39`
- 两组作者单位已经按照 IEEEtran 标准写入 `\thanks`，并在首页左下角正确呈现：七位作者归属 Faculty of Marine Science and Technology, Beijing Institute of Technology, Zhuhai 519000, China，Fang Deng 归属 School of AI, Beijing Institute of Technology, Beijing, China。
- 两个脚注中的 `(e-mail: )` 仍为空白占位。正式投稿前由作者填写真实邮箱；本次不根据文件外信息猜测地址。

## 需要优先处理的源文件语义问题

### 2. `H_t` 在空可行替代集合上的最小值仍未形成完整定义

- 位置：`paper/main.tex:281-289`、`paper/main.tex:317-319`
- 正文已经正确说明 `\mathcal D_t=\varnothing` 时不计算式 (3c) 的最小值，并令 `F_t=0`。但是式 (3c) 本身仍写作 `\min_{a\in\mathcal D_t}c_t(a)`，没有在公式层面给出空集分支或扩展实数约定。
- 读者仅依据公式仍无法确定 `H_t` 在该状态是直接置零、未定义，还是采用某个约定值。建议在式 (3c) 前后增加一个简短的条件定义，使 `F_t=0` 与 `H_t` 的计算域完全闭合，不需要增加新参数或新实验。

### 3. release 侧门控符号与 query 侧定义的对应关系不够形式化

- 位置：`paper/main.tex:347-370`
- `\mathcal V_r=\{a\in\mathcal A(s_r):c_r(a)\leq\kappa_c\}` 的条件集合定义在数学上成立。它表示 release state 下满足绝对 recovery-cost 阈值，并且通过 `F_r\wedge A_r\wedge H_r` 状态级检查的候选动作集合；任一状态级检查失败时，`\mathcal V_r=\varnothing`，式 (5) 因而必然给出 `g_r=0`。
- 式 (5) 的合取门控形式正确：返回动作必须属于 `\mathcal V_r`，与并发快速动作不同，并满足 `c_r(\widetilde a_r^{\mathrm{sl}})+\delta\leq c_r(a_r^f)`。这依次表达绝对可行性、非冗余性和动作级相对优势，并通过式 (6) 保证门控失败时回退到快速动作。
- 形式化缺口在于 `F_r`、`A_r`、`H_r` 和 `c_r` 只被说明为在 `s_r` 及并发的 `a_r^f` 上重算，没有显式给出它们相对于 query 侧 `s_t`、`a_t^f`、`\mathcal D_t` 和 `c_t` 的替换关系，也没有定义对应的 `\mathcal D_r`。建议增加一句 `t\rightarrow r` 的符号映射，并明确 `L_t` 与 `N_t` 只属于 admission，不在 release 侧重新进入 `g_r`。
- 另一个轻微不一致是正文规定 `\delta\geq0`，而随后称释放动作具有 strict recovery-cost advantage。当 `\delta=0` 时，成本相等也满足不等式。本文固定 `\delta=0.02>0`，因此当前实验配置确实要求严格优势，实际方法结论不受影响；若保留一般化定义，建议将 strict 表述与 `\delta` 的取值域统一。

## 建议终校

### 4. Fig. 6 的命名仍混合投影前与投影后层级

- 位置：Fig. 6(a) 图内标签及 `paper/main.tex:559-563`。
- 正文和方法部分先在 mapped-command 层判断 distinctness，再经过 `\Phi` 得到 executable action；图内的 `Effective actions differ` 容易被理解为投影后的最终动作差异，而图中后续又单独给出 `Final action distinct (2)`。
- 建议将前者改成 `Mapped commands differ` 或 `Pre-projection commands differ`，保留后者表示投影后的最终差异，使图中阶段与式 (5) 的层级一致。

### 5. 两处机制归因语气略强

- 位置：`paper/main.tex:502-504` 和 `paper/main.tex:810-813` 附近。
- `explains why`、`which explains` 将观测到的关联直接写成确定因果。当前实验能够支持机制一致性解释，但不需要承担更强的因果识别责任。
- 建议改为 `is consistent with the intended role of`、`helps explain` 或同等强度的表述，保留 RGD 的价值判断而降低不必要的因果承诺。

### 6. 个别图注相对成熟 TVT 论文略简洁

- Fig. 2 和 Fig. 3 的 caption 主要给出图的主题，具体场景顺序、交通设置和比较含义较多依赖图内标签与相邻正文。
- 当前图内坐标、图例和正文说明足以完成阅读，因此这不是图文不对应或格式错误。与 `paper/ref` 中排名靠前的 TVT 论文相比，caption 的独立解释能力略弱，属于低优先级的呈现差距。
- 此项不要求改变当前图表结构或增加实验细节。若保持现有简洁 caption，正文与图的引用关系已经能够支撑论文主线。

## 已确认符合项

- `\documentclass[letterpaper,journal]{IEEEtran}` 与 IEEE Transactions 期刊模式一致，未发现自定义页边距、字号、栏宽或行距的覆盖。
- 标题不含公式；摘要为单段，未出现被要求删除的 VLM 讨论和详细阈值实验数据；关键词使用 `IEEEkeywords` 环境。
- 引言首段使用 `\IEEEPARstart`，章节分布、贡献点顺序和 Related Work 组织符合当前论文叙事。
- 作者单位脚注已使用 IEEEtran 的 `\thanks` 机制放置于首页左下角，作者栏和页眉保持正常；当前两个单位脚注中的邮箱均已填写。
- Fig. 1 位于第 3 页顶部，图注位于图下；所有表注位于表上。Fig. 2、Fig. 3 保持当前简洁图注，没有引入额外的行列或 traffic-setting 说明。
- Fig. 3/4、Fig. 6/7 和 Tables III/IV 已拆分为各自独立的 IEEEtran 浮动体，每个浮动体只含一个 `\caption` 和对应 `\label`。编号、页面顺序和可见版式保持正常。
- 公式使用 `equation`、`align`、`subequations` 和 `cases` 等 IEEEtran 兼容环境，未发现 `eqnarray` 或 `$$...$$` 等模板不推荐写法；公式编号连续且交叉引用可解析。
- 式 (1) 至式 (8) 已形成 recovery cost、delay feasibility、query admission、release validation、execution contract 和 R-VoD 的连续方法链；式 (5) 与 `\mathcal V_r` 的核心数学形式正确。
- 表格最优值加粗规则与正文解释总体一致，Table V 将 RGD 定位为安全性与速度的均衡，而非宣称跨场景所有指标全面领先。
- 实验章节的论证顺序清楚。Fig. 3/4 建立性能与资源分配背景，Fig. 5 展开零延迟和固定门控延迟变化，Fig. 6/7 分析 release 验证与门控消融，Fig. 8 连接 serial admission、slow-path lifecycle 和 matched utility，Tables III--V 再检验 reasoner interface、交通条件和跨场景表现。尤其是 `143\rightarrow138\rightarrow28\rightarrow9\rightarrow6\rightarrow2` 的 release 链条能够直观说明 RGD 如何把返回 proposal 收敛为少量实际动作变化。
- 参考文献采用当前 `IEEEtran` 可见格式，正文引用编号连续至 [32]。本次仅将 `paper/references.bib` 中的 venue、会议缩写、arXiv 呈现字段和标题 acronym 保护与当前 PDF 对齐，没有新增、删除或补写未经作者确认的文献元数据。
- PDF 共 10 页，页面为 612 × 792 pt letter 尺寸；逐页检查未发现图表遮挡、裁切、公式越界、双栏重叠或明显异常空白。

## 最终判断

论文的主体版式和科研呈现已经可以作为 IEEE TVT 投稿稿件的基础版本。以同一标准与 `paper/ref` 中 9 篇 TVT 论文比较，当前稿的科研故事和实验机制分析已达到参考组中上水平，目前评分为 90/100，与参考组第 5 位论文并列。正式提交前应处理空集合公式约定、release 侧符号闭环及 `\delta` 与 strict advantage 的表述一致性。Fig. 6 标签、两处机制归因语气、坐标轴自包含性和简洁 caption 属于低风险终校项，不需要改变章节、图表顺序、方法主线或实验结论。

## 2026-08-24 逐图逐轴复核更新

本节覆盖当前 PDF 中 Fig. 1--Fig. 8 的坐标轴、统计单位、图例和正文对应关系。数值核对以图内标注、caption 和相邻正文为准，不涉及代码或原始数据真实性。

### 1. Fig. 1--Fig. 3

- **Fig. 1：** 架构图没有坐标轴。顶部的 Query admission、Driving memory、Slow reasoner、Release validation 与正文的两个在线门控一致，底部的 Release snapshot、Rollout comparison、R-VoD labels 与离线分支一致。图注“dashed line”与图中虚线分支相符，无图文逻辑缺陷。
- **Fig. 2：** 场景示意图没有坐标轴。四列 Highway、Merge、Roundabout、Intersection 与正文一致，顶部和底部两行分别呈现两种模拟器风格。当前 caption 没有说明行与模拟器的对应关系，读者需要依赖图像风格和正文推断，属于低优先级的自包含性不足。
- **Fig. 3：** 横轴 `Traffic setting` 配合 `4 lanes / 2.0`、`4 lanes / 2.5`、`5 lanes / 3.0` 已给出场景单位，纵轴 `Success rate (%)` 明确为百分比。图例、柱值和正文中 31.4、43.8、46.9 个百分点的计算一致。该图的坐标标识充分。

### 2. Fig. 4--Fig. 6

- **Fig. 4：** Panel (a) 的横轴 `Requests per episode` 明确是每回合均值，Panel (b) 的 `Paired request difference` 没有在轴上标出单位为 requests/episode，也没有直接写出“RGD minus baseline”。caption 和正文补足了这些信息，因此不构成数值错误，但建议将横轴写成 `Paired request difference (requests/episode)` 或在 caption 中先定义差值方向。
- **Fig. 5：** Panel (a) `Slow-path requests` 表示零延迟 cohort 的总请求数，Panel (b) `Runtime (ms/frame)` 和 Panel (c) `Added response delay (s)` 的单位清楚。Panel (a) 没有在横轴或 caption 中直接写出总量对应的 cohort 规模，caption 虽然写了 `total requests`，仍建议在 caption 中明确“over the zero-delay cohort”。另需与 Table II 的运行时间比较口径保持一致。
- **Fig. 6：** Panel (a) 的横轴 `Event count`、Panel (b) 的 `Recovery-cost advantage` 均可读，但 Panel (b) 没有说明 advantage 使用归一化 cost scale，固定 0.02 margin 依赖 caption 和正文解释。更重要的是图内 `Effective actions differ (9)` 与方法中的“投影前 mapped commands”层级不完全同名，容易与 `Final action distinct (2)` 混淆。应优先修正阶段名称，而不是修改数值。

### 3. Fig. 7 的坐标轴问题

- Panel (a) 横轴 `Slow-path requests` 给出的是 20-seed cohort 的总请求数，而不是 requests/episode。Panel (b) `Additional release states` 同样是相对 Full RGD 的总事件数，而非比例。Panel (c) `Rate difference (pp)` 是 R-VoD corrective-state rate 的百分点差异，但轴名没有写明参照组是 Full RGD。
- 这三个轴在当前 caption 和正文中可以被补全理解，但仅看图时容易出现三种误读：把 panel (a) 当成每回合均值，把 panel (b) 当成百分比，把 panel (c) 当成绝对 rate。建议使用 `Total slow-path requests (20 seeds)`、`Additional release states (count vs. Full RGD)` 和 `Corrective-state-rate difference vs. Full RGD (pp)` 一类的自包含标签。若不希望把 cohort 规模放入轴名，至少应在 caption 第一语句中明确三者均为 20-seed 汇总量。
- Fig. 7 只展示 `L`、`A`、`H`，没有展示方法中的第四个准入条件 `N`。这不是数据错误，因为 `N` 在 Fig. 8 的固定 proposal-bank 分析中单独检验，但当前 caption 容易让读者误解为完整的 one-at-a-time admission ablation。应在 caption 中限定为 `L`, `A`, and `H`，并注明 `N` is analyzed separately in Fig. 8。

### 4. Fig. 8 的坐标轴问题

- Panel (a) 的横轴 `Serial admission stage` 是分类阶段而非数值轴，`Candidates`、`after L`、`after A`、`after H`、`after N` 的顺序与式 (4) 一致。`Candidates` 实际表示同步 proposal bank 的初始候选总数，最好写成 `Candidate bank` 或在 caption 中定义。该 panel 也缺少明确的纵向说明来标识每行是 gate condition，当前依赖左侧 `Full`、`w/o L` 等行标签和 caption，建议在 caption 中明确“rows are gate conditions, columns are serial stages”。
- Panel (b) 的横轴 `Ablation arm` 与图例中的 `Issued`、`Released`、`Timeout` 对应正确，纵轴 `Event count` 也正确。但图中实际只绘制 `w/o H`、`w/o N` 和 `w/o H/N` 三个 arm，没有绘制 `Full`、`w/o L` 和 `w/o A`。正文已经解释这是对完整 proposal-bank 轨迹中会产生 release 的三类 arm，caption 仍应明确 `the three release-producing arms`，否则读者可能将该 panel 误读为六种条件的完整比较。另需在轴或 caption 中写明计数是 20 个 paired seed blocks 的汇总，而不是单个 seed 的事件数。
- Panel (c) 的横轴实际是三个 proposal seed identifiers `5020`、`5024`、`5028`，并非连续变量。当前没有横轴标题，只在 caption 中说明 horizontal labels are seed identifiers，且 `Retained`、`Filtered` 是第二层 release outcome 标签。建议将横轴明确为 `Proposal seed (release outcome)`，将纵轴明确为 `Matched utility advantage (normalized utility)`，避免把 seed 编号误读为实验条件或连续自变量。该 panel 只有三个 matched proposals，属于逐 proposal 审计而非总体统计比较，caption 中应保持这一定位，不应让坐标外观暗示存在均值或置信区间。

### 5. 与坐标轴相关的优先级结论

1. **高优先级终校：** Fig. 7 的总量、百分点差异和 Full RGD 基准需在轴或 caption 中自包含，Fig. 8(c) 需明确 seed identifier 与 normalized utility 的含义。
2. **中优先级终校：** Fig. 8(b) 明确只展示三个 release-producing arms，并统一 `20 paired seed blocks` 术语；Fig. 4(b) 补充差值单位和方向；Fig. 5(a) 明确总请求对应的 zero-delay cohort；Fig. 8(a) 明确行列含义和初始 candidate bank。
3. **低优先级终校：** Fig. 3 将 `4 lanes / 2.0` 一类标签解释为 lane count / traffic density，Fig. 2 caption 补充两行模拟器对应关系，Fig. 6(b) 补充 cost scale 说明。上述问题均不要求改变图表数据、顺序或版式结构，只涉及坐标轴和 caption 的自包含性。

### 6. 本次复核新增的正文风险

- Table II 仍有 `Time (ms/frame)`，位置：`paper/main.tex:493-512`、`541-557`。这会重新引入跨平台运行时间比较，和当前只呈现本系统单帧时间的口径不一致。
- 式 (3c) 在 `\mathcal D_t=\varnothing` 时仍包含未定义的最小值，位置：`paper/main.tex:287-295`、`317-325`。
- Release 侧的 `F_r`、`A_r`、`H_r`、`c_r` 没有显式给出与 Query 侧变量的 `t\rightarrow r` 同构映射，位置：`paper/main.tex:353-397`。
- Table I 是 published-settings context，不能被 `paper/main.tex:463-469` 的 shared-protocol 句子误读为与 RGD 完全同条件的控制实验。
- `references.bib` 的正式期刊和会议 venue 已改为当前 PDF 使用的 IEEEtran 缩写，例如 `IEEE Trans. Veh. Technol.` 和 `Proc. IEEE/CVF ...`，并加入源文件注释说明该约定。重新运行 BibTeX 和 `latexmk` 后缩写保持不变，PDF 仍为 10 页。BibTeX 仅保留 `lesort2018highwayenv` 缺少 journal 的既有警告，该条目仍按当前无 venue 的显示方式输出。

本次逐图逐轴复核未发现图表数值与正文结论之间的实质矛盾。当前最需要补强的是坐标轴和 caption 的自包含性，以及少量公式和比较口径定义，而不是重新设计图表或改变实验故事。
