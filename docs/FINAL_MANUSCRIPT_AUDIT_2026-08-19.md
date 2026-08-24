# RGD-Driver 最终稿件审查报告（IEEE TVT 模板与论文呈现）

- 审查日期：2026-08-24
- 审查对象：`paper/main.tex`、`paper/main.pdf`、IEEE 模板目录 `paper/IEEE-Transactions-LaTeX2e-templates-and-instructions`，以及 `paper/ref` 中的 IEEE 论文版式
- 审查范围：IEEEtran 源文件语义、作者栏、页眉、摘要与关键词、章节结构、公式、图表浮动体、参考文献呈现、PDF 版面
- 明确排除：代码实现、运行配置、原始实验数据、可复现性、文献真实性核验和正文未要求的内容重写

## 总体结论

当前稿件的 PDF 呈现已经符合 IEEE Transactions 双栏论文的主体视觉规范，整体达到 `paper/ref` 中部分 IEEE Transactions on Vehicular Technology 论文的基本呈现水准。`IEEEtran` 的 `letterpaper,journal` 模式、10 页 letter 页面、正文双栏、摘要与关键词位置、首段 drop cap、图表编号、公式编号、页眉、IEEE 引用样式和参考文献连续编号均正常。

本次检查确认：32 个正文引用与 32 个 `bibitem` 一一对应，没有未定义引用或交叉引用；PDF 编译成功，没有图表裁切、公式越界、正文重叠或明显破坏阅读的空白。当前稿件仍属于 **Minor Editorial Revision**，主要剩余问题是作者元数据完整性及两处方法表达的严格性。

## 必须修正

### 1. 作者单位和作者脚注信息缺失

- 位置：`paper/main.tex:31-32`
- 当前作者栏已正确使用作者间空格、`and` 连接和 `\IEEEmembership{Fellow,~IEEE}`，页眉 `Xie et al.: RGD-Driver` 也符合 IEEEtran 的 running-head 写法。
- 但 `\author{...}` 后没有任何 `\thanks{...}` 作者脚注，因此 PDF 中没有作者所属单位、城市国家、通信作者或邮箱信息。IEEE 模板的作者示例明确将 affiliations 放入 `\thanks`，当前稿不能视为完整的非匿名 IEEE 期刊作者元数据。
- 本次未根据文件外信息猜测单位和邮箱。提交前应由作者补入真实 affiliation；若有通信作者，应在相应脚注中明确标识。

## 需要优先处理的源文件语义问题

### 2. `H_t` 在空可行替代集合上的最小值仍未形成完整定义

- 位置：`paper/main.tex:281-289`、`paper/main.tex:317-319`
- 正文已经正确说明 `\mathcal D_t=\varnothing` 时不计算式 (3c) 的最小值，并令 `F_t=0`。但是式 (3c) 本身仍写作 `\min_{a\in\mathcal D_t}c_t(a)`，没有在公式层面给出空集分支或扩展实数约定。
- 读者仅依据公式仍无法确定 `H_t` 在该状态是直接置零、未定义，还是采用某个约定值。建议在式 (3c) 前后增加一个简短的条件定义，使 `F_t=0` 与 `H_t` 的计算域完全闭合，不需要增加新参数或新实验。

### 3. release 侧门控符号与 query 侧定义的对应关系不够形式化

- 位置：`paper/main.tex:347-370`
- `F_r`、`A_r`、`H_r` 和 `c_r` 被说明为在 `s_r` 及并发的 `a_r^f` 上重算，但没有明确给出它们相对于 query 侧 `s_t`、`a_t^f`、`\mathcal D_t` 和 `c_t` 的替换关系。
- 当前叙述足以理解算法意图，但严格审稿时仍需要读者自行推断 query-to-release 的状态、动作域和代价接口迁移。建议增加一句符号映射说明，并明确 `L_t` 与 `N_t` 只属于 admission，不在 release 侧重新进入 `g_r`。

## 建议终校

### 4. Fig. 6 的命名仍混合投影前与投影后层级

- 位置：Fig. 6(a) 图内标签及 `paper/main.tex:559-563`。
- 正文和方法部分先在 mapped-command 层判断 distinctness，再经过 `\Phi` 得到 executable action；图内的 `Effective actions differ` 容易被理解为投影后的最终动作差异，而图中后续又单独给出 `Final action distinct (2)`。
- 建议将前者改成 `Mapped commands differ` 或 `Pre-projection commands differ`，保留后者表示投影后的最终差异，使图中阶段与式 (5) 的层级一致。

### 5. 两处机制归因语气略强

- 位置：`paper/main.tex:502-504` 和 `paper/main.tex:810-813` 附近。
- `explains why`、`which explains` 将观测到的关联直接写成确定因果。当前实验能够支持机制一致性解释，但不需要承担更强的因果识别责任。
- 建议改为 `is consistent with the intended role of`、`helps explain` 或同等强度的表述，保留 RGD 的价值判断而降低不必要的因果承诺。

## 已确认符合项

- `\documentclass[letterpaper,journal]{IEEEtran}` 与 IEEE Transactions 期刊模式一致，未发现自定义页边距、字号、栏宽或行距的覆盖。
- 标题不含公式；摘要为单段，未出现被要求删除的 VLM 讨论和详细阈值实验数据；关键词使用 `IEEEkeywords` 环境。
- 引言首段使用 `\IEEEPARstart`，章节分布、贡献点顺序和 Related Work 组织符合当前论文叙事。
- Fig. 1 位于第 3 页顶部，图注位于图下；所有表注位于表上。Fig. 2、Fig. 3 保持当前简洁图注，没有引入额外的行列或 traffic-setting 说明。
- 公式使用 `equation`、`align`、`subequations` 和 `cases` 等 IEEEtran 兼容环境，未发现 `eqnarray` 或 `$$...$$` 等模板不推荐写法；公式编号连续且交叉引用可解析。
- 表格最优值加粗规则与正文解释总体一致，Table V 将 RGD 定位为安全性与速度的均衡，而非宣称跨场景所有指标全面领先。
- 参考文献采用当前 `IEEEtran` 可见格式，正文引用编号连续至 [32]。本次审查未修改 `paper/references.bib`，也未补写未经作者确认的文献元数据。
- PDF 共 10 页，页面为 612 × 792 pt letter 尺寸；逐页检查未发现图表遮挡、裁切、公式越界、双栏重叠或明显异常空白。

## 最终判断

论文的主体版式和科研呈现已经可以作为 IEEE TVT 投稿稿件的基础版本。正式提交前，优先补齐作者单位和脚注信息，并处理空集合公式约定及 release 侧符号闭环。Fig. 6 标签和两处因果语气属于低风险终校项，不需要改变章节、图表顺序、方法主线或实验结论。
