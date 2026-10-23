# Early vs Late Fusion 实验归档

**2026-10-23 整理入口：** [研究总览与阅读索引](RESEARCH_INDEX.md) · [证据核对与待办](report/证据核对与待办.md) · [整理前文件校验清单](FILE_MANIFEST.tsv)。合成日志和首轮 JSON 已核对；部分后续主结果缺少归档输出，本次未重跑实验。

**研究问题：** 为什么多模态心理学研究中，early fusion 常常优于 late fusion？early 的优势能否作为"模态间交互效应"存在的证据？

**归档日期：** 2026-10-19
**报告：** [`report/多模态融合策略实验报告.md`](report/多模态融合策略实验报告.md)

---

## 一、核心结论

1. **"early > late" 不能作为交互效应存在的证据。** 在已知无任何交互效应的合成数据上，early 依然 10/10 次胜过 late（领先 0.062–0.215 QWK，取决于单模态模型复杂度）。
2. **假设空间限制是待进一步验证的机制解释。** 原报告记载测试标签拟合的组合器仍落后 early 0.062 QWK（9/10 次）；对应实现与输出未完整归档，且该诊断不能代表所有 late 方法的上界。
3. **DAIC-WOZ 的原报告记载线性 early 优于所测试的 MLP。** QWK 为 0.213 与 0.154；对应主实验需补档核验，该比较不能排除真实交互。
4. **CCAC_26 被原研究排除为主实验数据。** 报告记载视频特征 86% 识别学校；校内平均音视频 QWK 为 0.007 / −0.012，LOSO 各校平均为 −0.006 / +0.020。相关脚本存在，最终输出待补档。

限定：第 1 条比较对象为学权重 late；加性线性设置中 stacking 为 0.664，略高于 early 的 0.661，不能概括为所有 late 方法都更差。

完整证据链、术语解释与局限见报告。

---

## 二、目录结构

```
early_vs_late/
├── README.md                      本文件
├── DATA_SOURCES.md                数据来源与引用要求（重要）
├── report/
│   └── 多模态融合策略实验报告.md    完整实验报告（含小白版摘要、术语表、全部数值）
├── code/
│   ├── ccac/                      CCAC_26 实验脚本（13 个）
│   └── daic/                      DAIC-WOZ 实验脚本（5 个）
└── results/
    ├── ccac/                      CCAC 的中间结果与 OOF 预测
    └── daic/                      DAIC 的汇总结果与合成实验日志
```

**未包含（按要求排除）：**

| 数据集 | 被排除的目录 | 原位置 |
|---|---|---|
| CCAC_26 | `train/`、`test/`、`_cache/` | `C:\Users\lenovo\Desktop\多模态心理学\CCAC_26\` |
| DAIC-WOZ | `data/`、`labels/`、`_cache/` | `D:\DAIC-WOZ\` |

归档中仅保留脚本、日志、汇总结果和 OOF 预测（这些是实验产物，不是原始数据）。

---

## 三、脚本说明

### 3.1 CCAC_26（`code/ccac/`）

按实验推进顺序排列，后一个依赖前一个的函数。

| 脚本 | 作用 |
|---|---|
| `_inventory.py` | 盘点特征目录：格式、维度、每例 clip 数、空向量 |
| `_build_cache.py` | 把 clip 级特征聚合为 subject×timepoint 级缓存 |
| `_fusion_probe.py` | 工具函数库（softmax 回归 / MLP / PCA / QWK / 分组 CV）+ 单模态基线 + 残差相关 |
| `_fusion_probe2.py` | 在分数基线上做增量融合 + 按班级 bootstrap |
| `_fusion_probe3.py` | 序数回归基线 + 加权 late 上界 + 交互效应 bootstrap |
| `_fusion_probe4.py` | 改目标为"变化量"（恶化 / 任何变化） |
| `_capacity_matched.py` | 容量匹配的 early/late 对照 + 分数趋势特征 |
| `_change_model.py` | Δ = y4 − y3 的岭回归建模与持续性锚定 |
| `_final_eval.py` | 5 次重复 GroupKFold × 2 种分组 + 置换检验 + 逐校拆分 |
| `_baseline_check.py` | 基线校准：l2 × 类别加权扫描、ridge λ 扫描、逐校标签结构 |
| `_robustness.py` | 按校分层 QWK、特征健全性、T3/T4 对照 |
| `_confound_check.py` | 站点混杂检验：特征能否识别学校 |
| `_loso.py` | leave-one-school-out 正式收口 |

**依赖关系：** `_fusion_probe.py` 是基础库；`_change_model.py` 从 `_capacity_matched.py` 导入 `score_feats`，未见反向导入。各 CCAC 脚本需保持在同一目录下。

### 3.2 DAIC-WOZ（`code/daic/`）

| 脚本 | 作用 |
|---|---|
| `_daic_cache.py` | 逐帧特征聚合为受试者级向量（8 套特征），构建标签 |
| `_daic_fusion.py` | 基础工具（PCA / softmax / MLP / QWK / AUC / 分层 CV） |
| `_daic_fusion2.py` | 嵌套 CV 学权重 + 双目标（5 级 / 二分类）+ stacking |
| `_daic_fusion_lean.py` | 精简运行版（5 seed，即时输出） |
| `_synthetic_power.py` | 合成数据功效校验（additive / interaction / null 三种机制） |

**依赖关系：** `_daic_fusion.py` 是基础库；`_daic_fusion2.py` 从它导入，`_synthetic_power.py` 和 `_daic_fusion_lean.py` 同时依赖这两个模块。

---

## 四、结果文件说明

| 文件 | 内容 |
|---|---|
| `results/ccac/_inventory.json` | 12 套特征目录的格式、维度、clip 数、空向量统计 |
| `results/ccac/_probe_results.json` | 第一轮单模态基线与融合对照的指标 |
| `results/ccac/_probe_oof.npz` | 第一轮全部 OOF 预测（按模型 × raw/pca/prob 分组） |
| `results/ccac/_probe2_oof.npz` | 第二轮（含分数基线）全部 OOF 概率 |
| `results/daic/_daic_fusion_results.json` | DAIC 首轮（非嵌套）融合结果 |
| `results/daic/_synthetic_power.log` | **合成数据功效校验的完整日志**（报告附录 B.1 的原始来源） |

---

## 五、复现说明

### 5.1 环境

- Python 3.14.6（Windows）
- numpy 2.5.1
- **未安装** pandas / scipy / scikit-learn
- 所有统计方法（softmax 回归、MLP、逻辑回归、岭回归、PCA、QWK、AUC、bootstrap、置换检验）均为 numpy 手写实现

### 5.2 路径依赖（重要）

所有脚本都以自身所在目录为根：

```python
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "train")     # CCAC
DATA = os.path.join(ROOT, "data")      # DAIC
```

因此依赖真实数据与缓存的 `code/` 副本**不能直接在归档目录里完成实验**。合成脚本不读取真实数据，安装 numpy 后可在项目根目录运行 `python code/daic/_synthetic_power.py`（本次未执行）。真实数据复现有两种方式：

1. **推荐**：把脚本复制回原数据目录（CCAC 脚本放回 `CCAC_26/`，DAIC 脚本放回 `DAIC-WOZ/`）。
2. 或者修改脚本顶部的 `ROOT` / `DATA` 路径，指向数据所在位置。

### 5.3 运行前需要的数据

| 数据集 | 需要放置的目录 | 来源 |
|---|---|---|
| CCAC_26 | `train/`、`test/` | 见 [`DATA_SOURCES.md`](DATA_SOURCES.md) |
| DAIC-WOZ | `data/`、`labels/` | 见 [`DATA_SOURCES.md`](DATA_SOURCES.md) |

### 5.4 运行顺序

**CCAC：**
```
_inventory.py → _build_cache.py → _fusion_probe.py → _fusion_probe2.py
→ _fusion_probe3.py → _fusion_probe4.py → _capacity_matched.py
→ _change_model.py → _final_eval.py → _baseline_check.py
→ _robustness.py → _confound_check.py → _loso.py
```
耗时：`_build_cache.py` 约 1–2 分钟；`_final_eval.py` 约 5–10 分钟；其余每个 1–5 分钟。

**DAIC：**
```
_daic_cache.py → _daic_fusion.py → _daic_fusion2.py（或 _daic_fusion_lean.py）
→ _synthetic_power.py
```
耗时：`_daic_cache.py` 约 5 分钟；`_synthetic_power.py` 约 10–20 分钟。

### 5.5 已知运行陷阱

1. **DAIC 的 BoAW/BoVW 文件有两种分隔符**（逗号和分号混用），`_daic_cache.py` 已做自动识别。
2. **高维特征 PCA 必须走 n×n Gram 矩阵路径**（`_daic_fusion.py` 的 `prep` 已修正）。若用 d×d 协方差矩阵，vgg16 的 4096 维会让单次运行超过 30 分钟。
3. **CCAC 深度音频特征时间分辨率仅 0.05 Hz**（60 秒片段 1–3 帧），不可用于时序建模；时序分析只能用 `audio_basic`（25 Hz）。
4. **CCAC 的 SCH_005 只有 1 人**，所有按学校划分的分析必须剔除。

---

## 六、数据来源

详细的许可与引用要求见 [`DATA_SOURCES.md`](DATA_SOURCES.md)。

- **CCAC_26**：中国情感计算大会数据集，来源 <https://ccacmaps.hai-lab.cn/>
- **DAIC-WOZ**：Distress Analysis Interview Corpus，需引用 Gratch et al. (LREC 2014) 与 Ringeval et al. (AVEC 2019)
- **合成数据**：本实验用 numpy 生成，无外部依赖

---

## 七、未归档的文件

以下文件属实验开始前已存在的工作文件，未纳入本归档：

- `CCAC_26/_analyze_ccac.py`
- `CCAC_26/_baseline_probe.py`
- `CCAC_26/_inspect_files.py`
- `CCAC_26/_quest_ceiling.py`
- `CCAC_26/_verify_dims.py`
- `CCAC_26/extract_all_7z.ps1`、`extract_run.log`
- `CCAC_26/.zwork/`

原始实验报告同时保留在 `C:\Users\lenovo\Desktop\多模态心理学\多模态融合策略实验报告.md`，归档内为同一份副本。
