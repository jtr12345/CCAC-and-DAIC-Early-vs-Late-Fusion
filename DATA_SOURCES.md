# 数据来源与引用要求

本实验使用两个真实数据集和一个合成数据集。**若你在任何论文、报告、学位论文或演示中使用本实验（尤其是其中的结果），必须按下列要求引用对应数据来源。**

---

## 一、DAIC-WOZ（Distress Analysis Interview Corpus — Wizard-of-Oz）

### 1.1 来源说明

DAIC-WOZ 由南加州大学创意技术研究所（USC Institute for Creative Technologies）发布，用于抑郁症自动检测研究。受试者与虚拟面试官进行半结构化访谈，包含音频、视频（面部特征）与转录文本。

本实验使用的版本包含 275 名受试者，特征是预先提取好的（eGeMAPS、MFCC、OpenFace AU、BoAW/BoVW、VGG16、DenseNet 等）。

### 1.2 引用要求（原文照录）

> **Remember to cite the references below if you produce a paper that in any way utilizes this database.**

**必须引用的文献：**

Gratch J, Artstein R, Lucas GM, Stratou G, Scherer S, Nazarian A, Wood R, Boberg J, DeVault D, Marsella S, Traum DR. The Distress Analysis Interview Corpus of human and computer interviews. In LREC 2014 May (pp. 3123-3128)

Ringeval, Fabien, Björn Schuller, Michel Valstar, Nicholas Cummins, Roddy Cowie, Leili Tavabi, Maximilian Schmitt et al. "AVEC 2019 Workshop and Challenge: State-of-Mind, Detecting Depression with AI, and Cross-Cultural Affect Recognition." In Proceedings of the 9th International on Audio/Visual Emotion Challenge and Workshop, pp. 3-12. ACM, 2019.

### 1.3 本实验的具体使用方式

- 标签：`labels/detailed_lables.csv` 中的 PHQ-8 八项条目，自行加总为总分
- 二分类阈值：PHQ-8 ≥ 10
- 五级严重度：0–4 / 5–9 / 10–14 / 15–19 / 20+
- 模态：音频（eGeMAPS、MFCC、BoAW）、视频（OpenFace AU、VGG16、DenseNet、BoVW）
- 样本：274 人（275 个目录中 1 人缺特征文件）
- **未使用**：原始 wav 音频、文本转录（因为实验只比较融合层级，需要固定编码器）

### 1.4 已知局限（引用结果时请一并说明）

- 单站点采集，虽排除了站点混杂，但可能限制外部效度
- 存在"虚拟面试官"混杂：面试官行为可能同时影响受试者表达与标签
- n=274 属小样本；五级严重度中最高级仅 10 人

---

## 二、CCAC_26

### 2.1 来源说明

CCAC_26 是中国情感计算大会（CCAC）相关竞赛数据集。比赛任务为：根据 T1、T2、T3 三个时间点的多模态数据，预测 T4 时间点的焦虑等级。

**来源：** <https://ccacmaps.hai-lab.cn/>

该网站同时说明数据来源、任务定义、评估指标与提交方式。使用前请在该网站确认最新的许可条款与引用要求。

### 2.2 本实验的具体使用方式

- 训练集：1527 名中学生 × 3 个时间点 = 4581 条 subject×timepoint 记录
- 标签：`train/labels.csv` 的 `t4_anxiety_level`（5 级：正常 / 轻度 / 中度 / 重度 / 非常严重）
- 特征：音频（18 维手工特征 + 4 套深度编码器）、视频（3 维手工特征 + 5 套深度编码器）、转录文本
- 结构：8 所学校 / 83 个班级
- 测试集 1909 人**无标签**，本实验未使用

### 2.3 关键说明：本实验最终**排除**了该数据集

CCAC_26 在本实验中被用作初始候选，但经过混杂检验后被排除，原因见报告第 4 章。若引用本实验对该数据集的结论，请一并引用以下事实：

- 视频特征识别学校的准确率为 **86.0%**（7 类随机基准 14.3%）
- 按学校分层后，音频与视频的 QWK 分别为 **0.007** 与 **−0.012**
- SCH_005 仅 1 人；SCH_006 的 311 人中 306 人 T4 为"正常"

**建议：** 任何使用 CCAC_26 的研究都应报告站点混杂检验，并采用 leave-one-school-out 评估。

---

## 三、合成数据

**来源：** 本实验自行生成，无外部数据依赖。

生成方式见 `code/daic/_synthetic_power.py`：

| 机制 | 生成规则 |
|---|---|
| additive | y ∝ 1.5·a₀ + 1.5·v₀ + 噪声 |
| interaction | y ∝ 2.5·(a₀·v₀) + 噪声 |
| null | y ∝ 纯噪声 |

样本量 n=274，特征维度 d=32，5 类均衡分箱，10 个随机种子。可直接复现，无需引用。

---

## 四、引用汇总（可直接复制）

如果上述两个数据集都需要引用：

```bibtex
@inproceedings{gratch2014daic,
  title={The Distress Analysis Interview Corpus of human and computer interviews},
  author={Gratch, Jonathan and Artstein, Ron and Lucas, Gale M and Stratou, Giota and Scherer, Stefan and Nazarian, Angela and Wood, Rachel and Boberg, Jill and DeVault, David and Marsella, Stacy and Traum, David R},
  booktitle={Proceedings of the Ninth International Conference on Language Resources and Evaluation (LREC 2014)},
  pages={3123--3128},
  year={2014}
}

@inproceedings{ringeval2019avec,
  title={AVEC 2019 Workshop and Challenge: State-of-Mind, Detecting Depression with AI, and Cross-Cultural Affect Recognition},
  author={Ringeval, Fabien and Schuller, Bj{\"o}rn and Valstar, Michel and Cummins, Nicholas and Cowie, Roddy and Tavabi, Leili and Schmitt, Maximilian and others},
  booktitle={Proceedings of the 9th International on Audio/Visual Emotion Challenge and Workshop},
  pages={3--12},
  year={2019},
  organization={ACM}
}
```

CCAC_26 的引用格式请以 <https://ccacmaps.hai-lab.cn/> 公布的要求为准。
