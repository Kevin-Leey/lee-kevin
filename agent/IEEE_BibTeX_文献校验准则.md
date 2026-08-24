# IEEE BibTeX 文献引用校验准则

## 1. 总原则
文献引用校验的核心不是让所有 BibTeX 条目“长得一样”，而是确保**引用对象正确、文献版本明确、元数据真实完整、BibTeX 类型正确、IEEE 最终排版一致**。任何文献都必须先确定实际引用版本，再从该版本的权威来源获取或核验 metadata；禁止把 preprint、会议版、期刊版的字段混合到同一个 BibTeX 条目中。最终参考文献格式应交由 IEEE 官方 bibliography style/模板处理，不应通过手工拼接最终 PDF 格式来实现“统一”。

## 2. 文献元数据权威来源优先级
对于 IEEE/正式出版论文，元数据校验优先级建议为：**正式出版页面/IEEE Xplore > Publisher 官方页面 > Crossref > 官方会议页面 > arXiv/机构仓储 > Google Scholar/Semantic Scholar 等聚合数据库**。IEEE Xplore 可直接下载 BibTeX，但下载后仍应检查 DOI、作者、标题、年份、卷期页码等关键字段；Google Scholar 等主要用于发现论文，不宜作为正式出版 metadata 的唯一依据。Crossref 主要用于 DOI 和 bibliographic metadata 交叉核验。

## 3. 引用版本选择规则
必须先判断论文处于哪种出版状态：**preprint、conference、early access、正式 journal version、accepted manuscript、version of record**。当正式出版版本已经存在时，通常优先引用 **Version of Record/正式出版版本**，因为其作者、标题、DOI、卷期、页码等 metadata 最完整稳定；不能因为存在更早的 arXiv 就继续默认引用 preprint。若论文尚未正式出版，则可引用当前可验证的 preprint。若存在 conference 版和 journal extension，两者属于不同正式出版物时可以分别引用，但应根据具体 scientific claim 选择对应版本，不能无意义地同时引用。

## 4. Preprint、会议版、期刊版的处理
如果论文只有 arXiv/preprint，应如实标记为 preprint，不得虚构 volume、issue、pages、正式 DOI 等信息。如果论文已经正式发表，通常将对应 preprint citation 更新为正式版本。若研究内容明确区分“最初 conference 版本”和“后续 journal 扩展版本”，可以分别引用。若正式版本只是 publication state 更新（例如 Early Access → volume/issue/pages），不要建立两个不同论文条目，而应更新同一出版物的 metadata。不同版本若作者列表、方法、理论或实验发生实质变化，不得简单把一个版本的字段覆盖到另一个版本上。

## 5. BibTeX 类型规范
期刊论文通常使用 `@article`，会议论文通常使用 `@inproceedings`，预印本可根据所用 bibliography workflow 使用 `@misc` 或适当的 preprint 类型；不要因为论文收录于 IEEE Xplore 就把 conference paper 错写成 `@article`。核心是**文献类型必须与实际出版物类型一致**。

### 期刊论文
```bibtex
@article{key,
  author  = {...},
  title   = {...},
  journal = {...},
  year    = {...},
  volume  = {...},
  number  = {...},
  pages   = {...},
  doi     = {...}
}
```

### 会议论文
```bibtex
@inproceedings{key,
  author    = {...},
  title     = {...},
  booktitle = {...},
  year      = {...},
  pages     = {...},
  doi       = {...}
}
```

### arXiv/preprint
```bibtex
@misc{key,
  author        = {...},
  title         = {...},
  year          = {...},
  eprint        = {xxxx.xxxxx},
  archivePrefix = {arXiv},
  primaryClass  = {...}
}
```
具体字段以目标 bibliography style 和实际来源 metadata 为准，不要机械套模板。

## 6. 必查字段
正式期刊/会议文献至少检查：**author、title、venue（journal/booktitle）、year、DOI**；若正式来源提供，还应检查 **volume、number、pages 或 article number**。正式出版且存在 DOI 时应优先保留 DOI。DOI 不允许猜测，必须从 IEEE Xplore、publisher 或 Crossref 等权威来源核验。推荐使用规范 DOI 链接 `https://doi.org/...`；BibTeX 中则按当前 bibliography workflow 的字段要求填写 `doi`。

## 7. Author 规范
作者必须与引用版本的正式 metadata 一致。BibTeX 作者应使用 `and` 分隔，例如：

```bibtex
author = {Zhang, Wei and Wang, Hao and Li, Yang}
```

不要把多个作者写成普通逗号分隔字符串。注意同一论文不同版本可能存在作者顺序、作者数量变化，必须以当前引用版本为准。

## 8. Title 大小写保护
BibTeX/bibliography style 可能自动处理标题大小写，因此模型名、缩写和专有名词需要必要时使用大括号保护，例如：

```bibtex
title = {A {VLM}-Based Framework for Autonomous Driving}
```

应特别保护 `LLM`、`VLM`、`VLA`、`GPT-4`、`GPT-4o`、`Qwen`、`LLaMA`、`LiDAR`、`BEV`、`3D` 等必须保持特定大小写的术语。不要为了模仿最终 PDF 直接把 journal/title 字段手工改成某种显示形式，应优先保留权威 metadata，并让 IEEE style 负责最终排版。

## 9. Journal/Conference 名称规范
BibTeX 中的 `journal`、`booktitle` 应优先采用正式出版源提供的名称，不要自行猜测或随意缩写。最终 IEEE reference 中的期刊/会议名称及缩写形式应由 IEEE 官方规范和 bibliography style 统一处理。不要通过手工缩写来追求“看起来一致”。

## 10. 版本一致性校验
同一引用条目内部必须满足：**title、authors、year、venue、DOI、volume/issue/pages** 都来自同一个出版版本。严禁出现以下混合：

- arXiv 标题 + journal DOI
- conference venue + journal volume/pages
- preprint year + later journal issue
- conference 作者列表 + journal DOI
- 一个版本的 pages + 另一个版本的 DOI

发现这种情况时，应重新从目标版本的官方页面获取完整 metadata，而不是局部修补。

## 11. Duplicate/版本去重规则
检索大量论文时，应主动识别 `arXiv → conference → journal` 的版本链。判断同一工作不能只看标题，还应综合 **title、作者、DOI、arXiv ID、venue、年份、论文内容**。如果正式版本已经存在，通常删除重复的 preprint citation；只有当论文明确讨论不同版本、内容存在实质差异，或需要同时指向公开 preprint 和正式版本时，才保留多个引用。

## 12. Early Access 规则
IEEE 论文可能经历 `Accepted → Early Access → Volume/Issue → Final publication metadata`。Early Access 与最终卷期通常属于同一个正式出版物，不应误认为两篇论文。Early Access 有正式 DOI 时可以正常引用；当最终 volume、number、pages/article number 出现后，应更新 BibTeX 为最终 metadata。

## 13. Reference Key 规范
BibTeX key 应保持全文唯一、可读、稳定，例如：

```text
wang2026xxx
liu2025xxx
zhang2025vlaplanning
```

避免 `paper1`、`aaa`、`newpaper` 等没有语义的信息。更换论文版本时，若仍是同一正式出版物的 metadata 更新，不必无意义地修改 key；若是不同正式出版物，应使用不同 key。

## 14. 最终 IEEE 格式校验
不要通过手工修改 `.bib` 来强行模拟最终 IEEE reference 样式。应使用 IEEE 官方模板和对应 bibliography style，然后重点检查：引用编号是否为 `[1]`、`[2]` 等；正文引用编号是否与 reference list 正确对应；作者、标题、venue、年份、卷期页码、DOI 是否准确；会议/期刊/preprint 类型是否正确；特殊大小写是否保留；同一文献是否重复；是否存在未引用或引用但未进入 bibliography 的条目。

## 15. 推荐的标准工作流
```text
发现论文
  ↓
确定要引用的具体版本（preprint / conference / journal）
  ↓
优先打开正式出版页面/IEEE Xplore
  ↓
下载或获取该版本官方 BibTeX
  ↓
用 DOI + Crossref 核对 metadata
  ↓
检查作者/标题/venue/year/volume/issue/pages/DOI
  ↓
检查是否存在更新的正式版本
  ↓
去除同一工作的无必要重复 preprint
  ↓
保护 LLM/VLM/VLA/GPT/Qwen/LiDAR/BEV 等大小写
  ↓
统一 BibTeX key 和字段风格
  ↓
交给 IEEE bibliography style 编译
  ↓
检查最终 PDF reference list 与原始 metadata
```

## 16. 自动校验时必须执行的硬性规则
1. **不得编造任何 bibliographic metadata。** DOI、volume、issue、pages、article number 等必须能够在权威来源中验证。
2. **不得混用版本字段。** 每条 BibTeX 必须对应一个明确的出版版本。
3. **正式发表版本优先。** 已存在 Version of Record 时，通常不要继续引用对应 preprint。
4. **不能仅凭标题判断是否同一论文。** 必须结合作者、DOI/arXiv ID、venue、年份和内容。
5. **不能只依赖 Google Scholar。** 聚合数据库用于发现和辅助，正式 metadata 优先使用 IEEE Xplore、publisher、Crossref。
6. **不能手工猜测 IEEE 缩写和最终排版。** 保留可靠 metadata，让 IEEE style 负责呈现。
7. **会议与期刊必须区分。** `@inproceedings` 不因进入 IEEE Xplore 而变成 `@article`。
8. **特殊术语必须保护大小写。** 尤其是 AI、VLM、LLM、VLA、GPT、Qwen、LiDAR、BEV 等。
9. **同一工作无必要不得重复引用 preprint + conference + journal。** 有明确 scientific reason 时才同时保留。
10. **最终检查必须基于编译后的 PDF。** 不能只检查 `.bib` 文件本身。

## 17. 权威参考入口
- IEEE Reference Guide: https://journals.ieeeauthorcenter.ieee.org/wp-content/uploads/IEEE-Reference-Guide.pdf
- IEEE Editorial Style Manual: https://journals.ieeeauthorcenter.ieee.org/create-your-ieee-journal-article/create-the-text-of-your-article/ieee-editorial-style-manual/
- IEEE Xplore Citation/Download Citation: https://ieeexplore.ieee.org/
- Crossref: https://www.crossref.org/

## 18. 一句话判定原则
> **先确定“我实际引用的是哪一个出版版本”，再从该版本的权威来源取得 BibTeX，并用 DOI/Crossref 交叉核验；正式版优先于对应 preprint，禁止跨版本拼接 metadata，最终格式交给 IEEE style 而不是手工修改。**
