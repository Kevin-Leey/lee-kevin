# RGD-Driver 论文 AI 化风格审查

日期：2026-08-26  
审查对象：`paper/main.tex` 当前英文稿  
审查范围：句式、措辞、连接方式、论断语气、词语重复和技术术语的自然度  
审查性质：只读风格审查，不修改论文正文

## 1. 审查口径

本次审查参考 `humanizer` v2.11.2 对 AI 写作痕迹的定义，并采用
`academic-paper-reviewer` v1.11.1 的证据定位和严重度记录方式。重点检查以下
模式：

- 固定句式和重复句首
- 过度对称的并列结构和人为凑成的三项或四项列表
- 抽象名词、空泛总结动词和过强的结果动词
- 连续使用的现在分词短语和 `while` 从句
- 被动语态是否掩盖了动作主体
- 过多的技术复合词、连字符词和模板化连接词
- 广告化、聊天机器人式、泛化式和无来源的表述

统计词频只用于定位潜在风格风险，不把正式学术词汇或算法术语直接判定为
AI 痕迹。所有判断均以正文当前内容为依据，没有引入新的事实、数据或引用。

## 2. 总体结论

稿件整体的 AI 化风险为 **低到中等，且集中在少数段落**。方法定义、公式、实验
数字和图表解释较为具体，正文没有发现聊天机器人问候、广告化形容词、无来源的
“专家认为”式表述、标题式小结或泛泛的未来展望堆叠。论文的主要风格风险来自
学术写作中常见但重复偏高的模板化组织方式，而不是内容空洞或事实表达不自然。

当前没有需要立即修复的严重风格问题，也不建议进行全文重写。若进行下一轮
微调，优先处理摘要、Related Work 的段末总结、实验结果解释和 Conclusion 中的
少数句子即可。技术术语 `release-aware`、`closed-loop`、`slow-path`、
`recovery-cost`、`matched-rollout` 和 `post-projection` 不应为了降低 AI 痕迹而
删除或改成不准确的普通表达。

## 3. 分项发现

| 编号 | 严重度 | 位置 | 观察 | 判断与建议 |
|---|---|---|---|---|
| A1 | 中 | `main.tex:56` | 摘要把问题、方法、两个门控阶段、R-VoD 和实验结论压缩到一个很长的连续段落，并连续使用多个四项列表。 | 属于摘要压缩造成的节奏问题，不是事实错误。若后续允许微调，可把最后一个结果句拆开，或将第二次完整列举改为“the four admission predicates”。 |
| A2 | 中 | `main.tex:56,101,888` 及 `474,499,509,605,644,670,817,872,902` | `This paper presents` 出现 3 次，多个实验段落连续以 `RGD` 开头。 | `This paper presents` 在摘要、引言和结论中属于 IEEE 常规结构，不应单独判为 AI。实验段落可在未来修改时偶尔改为“the gate”“the evaluation”或直接以动作开头，以减少机械节奏。 |
| A3 | 中 | `main.tex:145,159,191,449,519,829,850,902` | 使用 `These studies establish`、`This progression improves`、`This separation prevents`、`This construction directly supports`、`The result demonstrates`、`the table therefore shows` 等指代性总结句。 | 这些句子有明确指代对象，但部分句子把上一段内容再次抽象成结论，带有 AI 常见的“先复述再拔高”节奏。优先检查 `829` 和 `850`，将总结改为直接描述机制或数据会更自然。 |
| A4 | 中 | `main.tex:896--903` | Conclusion 使用 `Comprehensive closed-loop experiments demonstrate` 和 `RGD establishes a principled, release-aware decision interface`。 | `Comprehensive`、`demonstrate`、`establishes`、`principled` 叠加后语气偏宣传式，是全文最明显的 AI 化和过度拔高组合。当前主张有正文支撑，但风格上可改为更具体的“Experiments show ...”和“RGD defines ...”一类句式。 |
| A5 | 低 | `main.tex:56,104,252,621` | `delay feasibility, action support, recovery-cost advantage, and scene demand` 四项门控条件在摘要、引言、方法和实验分析中反复完整出现。 | 这些是方法的真实组成，不应删除。重复出现时可使用“the four admission predicates”或“the admission checks”，避免每次重新展开。 |
| A6 | 低 | `main.tex:56,512,650,656,683,845,896` | 多处使用 `while` 加现在分词或并列动词，例如 `while preserving ... and ... filtering ...`、`while mean distance remains ...`。 | 这些从句大多承载真实的比较关系，不属于空洞 filler。问题是连续出现后节奏趋同。后续精修时可把一两处改成独立句，但不需要系统性删除。 |
| A7 | 低 | `main.tex:605--623,670--742` | 实验分析反复使用 `retains`、`removes`、`provides`、`supports`、`distinct` 等固定动词。当前词频约为 `retain*` 21、`support*` 20、`distinct*` 13。 | 这些词与 RGD 的机制直接相关，不能简单替换。可通过交替使用“passes”“is rejected”“maps to”“leaves”等更具体的动作词，降低机械重复。 |
| A8 | 低 | `main.tex:145--165,227--230` | Related Work 中出现概括性段末句：`These studies establish ...`、`This progression ...`、`Across these research directions ...`。 | 文献综述需要综合句，但这些句式偏通用。更自然的写法是直接指出已有工作的共同接口和本文缺口，减少“研究方向已充分发展”的抽象总结。 |
| A9 | 低 | `main.tex:56,79,91,499,509,829,902` | `demonstrate*`、`establish*`、`effective`、`meaningful`、`favorable` 等词在摘要、结果和结论中承担较强评价功能。 | 词语本身并非错误，风险在于它们与“best”“substantially”“principled”等词连续出现时会形成宣传式语气。建议只保留有明确图表或公式支撑的强动词，并让其后紧跟具体机制或指标。 |
| A10 | 通过 | 全文 | 未发现 `I hope this helps`、`let me know`、`here is an overview` 等聊天机器人残留，也未发现 `vibrant`、`tapestry`、`testament`、`landscape` 等非技术广告化词汇的集中使用。 | 这是稿件的优点。无需为了“像人类”而加入第一人称、口语化旁白或主观评价。 |
| A11 | 通过 | 全文 | 未发现无来源的 `experts argue`、`observers note`、`industry reports` 等模糊来源，也未发现标题中每个实词首字母大写、表格前使用粗体小标题或表情符号等格式痕迹。 | 符合 IEEE 科研文本习惯。 |
| A12 | 通过 | 全文 | 未发现 em dash、en dash 或 ` -- ` 作为叙述性破折号使用。正文中的连字符主要出现在技术复合词。 | `R-VoD`、`closed-loop`、`release-aware`、`recovery-cost`、`slow-path` 等属于稳定术语，应保留，不应机械拆开。 |
| A13 | 低 | 全文方法段，尤其 `main.tex:235--446` | 方法部分大量使用 `is defined`、`is evaluated`、`are calibrated`、`is retained` 等被动结构。 | 这里的被动语态符合公式和算法定义的 IEEE 写法，且多数句子关注变量而非作者，不构成 AI 痕迹。只有在实验结果句中遮蔽动作主体时，才建议改成主动语态。 |
| A14 | 低 | `main.tex:742` | `Panels (a), (b), and (c) establish a complete mechanism chain` 将三个图板的关系总结为“complete mechanism chain”。 | 图板确实按 admission、projection 和 release validation 组织，但 `complete` 是泛化性评价。若后续收紧语气，可改为直接陈述三个 panel 分别对应的机制。 |

## 4. 分章节判断

### 摘要

摘要的科研叙事完整，问题、缺口、RGD、R-VoD 和实验结论均有明确位置。主要
风格问题是句子过长和条件列表重复，而不是词汇不专业。`At query time` 与
`At release` 构成清楚的时序对照，属于方法核心，不应删除。最后一句同时承载
资源减少、任务保持和 release filter 三个结论，阅读负担较大，是最值得优先
微调的位置。

### Introduction 与 Related Work

引言中的场景例子和 query-state 到 release-state 的转折较自然，未见明显的
聊天机器人式铺垫。Related Work 的问题主要在段末总结句较抽象，特别是
`These studies establish`、`This progression` 和 `Across these research directions`
形成相似的综述收束模板。它们没有改变科研逻辑，但可在投稿前改成更直接的
“已有方法解决什么，本文接口缺什么”。

### 方法

方法章节是全文最不 AI 化的部分。公式、变量、集合和流程均以定义为中心，
被动语态和 `where` 从句属于必要的技术表达。`R-VoD does not enter the online
gate`、`Since ... by construction` 等边界句明确，不应为了追求口语化而删除。

### 实验与讨论

实验章节的数字、事件漏斗和图表指向具体，整体不像实验日志或聊天式解释。
需要留意的是，`explains why`、`therefore shows`、`establish a complete mechanism
chain` 等句子有时把描述性结果写成了机制证明。它们更接近“AI 化的归纳式
结论”，而不是实验事实本身。后续如需精修，应让句子先说明观察到的变化，再
说明该变化与 RGD 机制一致，而不是直接宣布机制已被证明。

### Discussion 与 Conclusion

Discussion 已经较短，限制和未来工作也没有套用大段泛化展望。Conclusion 的
结构清楚，但最后两句的 `Comprehensive`、`demonstrate`、`principled` 组合是
全篇最明显的宣传式收束。建议把强评价词减到一个，并保留“evaluated proposal
stream”这一范围限定。

## 5. 优先级建议

### 建议优先处理

1. 检查 `main.tex:896--903` 的 Conclusion 强结论组合，优先降低泛化性形容词和
   抽象总结动词的叠加。
2. 在 `main.tex:145--165` 和 `227--230` 中减少通用的 Related Work 段末总结，
   直接连接到“延迟 proposal 如何获得 actuation authority”这一缺口。
3. 对 `main.tex:56,104,252,621` 的四项门控条件，在首次完整定义后使用简短指代，
   避免重复展开同一列表。

### 可选微调

1. 在实验结果段落中交替使用具体动作词，降低 `retain`、`support`、`distinct`
   的连续重复。
2. 将少量连续的 `while` 从句改为独立句，改善摘要和结果段落的节奏。
3. 在 `main.tex:742` 将“complete mechanism chain”改为三个 panel 的直接对应，
   使图表解释更像证据说明而不是结论口号。

### 不建议处理

1. 不要删除 `release-aware`、`closed-loop`、`recovery-cost`、`slow-path`、
   `matched-rollout` 等技术复合词。
2. 不要把方法中的被动定义句全部改成主动句，也不要为追求“人类化”加入第一人称
   或口语化旁白。
3. 不要删除 `paired sign-flip tests` 等统计术语中的必要限定词。

## 6. 最终裁定

当前稿件存在局部、可控的 AI 化句式痕迹，但没有形成影响 IEEE TVT 阅读体验或
科研可信度的系统性问题。技术内容的具体性、公式边界和实验数字已经抵消了
大部分模板化风险。建议只做上述局部微调，不进行全文人类化重写，也不改变论文
现有章节结构、公式、图表和核心结论。

本次审查未修改 `paper/main.tex`、`paper/main_zh.md`、图表或参考文献。
