# Group 2：感知图像质量筛选（中文说明）

> 项目：Quality-Aware Synthetic Data for Road Scene Understanding  
> 小组：Group 2 — Perceptual Image Quality  
> 成员：Yingchao Li、Jiaying Ping  
> 当前版本：contrast-gradient fallback 第一版，双版本整合并复核（2026-11-05）
# Group 2 核心文件说明

| 文件 | 价值 |
|---|---|
| `results/fallback-v1/scores.jsonl` | 78,840 张图片的正式质量分数，核心实验结果 |
| `results/fallback-v1/perceptual_quality_budget_0.25x.jsonl` | 最终选出的 548 张图片，最重要的训练子集 |
| `src/group_2/perceptual_quality.py` | 评分、归一化和 Top-548 选择算法 |
| `src/group_2/cli.py` | 老师和队友可以直接运行的命令行入口 |
| `tests/test_group_2.py` | 证明确定性、schema、损坏图片处理和 548 张选择正确 |
| `README.md` | 正式英文方法说明 |
| `scripts/build_manifest.slurm` | 在 HPC 上生成 manifest 的可复现脚本 |
| `scripts/score_quality.slurm` | 在 HPC 上执行全量质量评分的脚本 |


## 1. 我们要解决什么问题

项目研究的问题是：在合成图数量固定的情况下，按照图像质量挑选合成训练图，是否比随机挑选更有利于语义分割？

所有选择方法都使用相同预算：

- 真实训练图：2,190 张；
- 合成图预算：`0.25 × 2190 = 548` 张；
- 分割模型：DeepLabV3+；
- 编码器：MobileNetV3 Large 0.75；
- 类别数：9；
- 正式实验种子：17、42、73。

老师提供两个共享基线：

- `real_only`；
- `random_budget_0.25x`。

Group 2 最终需要生成自己的固定子集：

```text
perceptual_quality_budget_0.25x
```

核心比较是：

```text
real_only
vs. random_budget_0.25x
vs. perceptual_quality_budget_0.25x
```

## 2. 当前方法

第一版采用仓库允许的 dependency-light contrast-gradient fallback。

处理流程如下：

1. 从 canonical manifest 中筛选 `eligibility.analysis=true` 的合成图；
2. 将图像解码并转换为灰度数组；
3. 使用灰度对比度（标准差）和局部梯度信息计算原始质量分；
4. 在所有成功评分的 analysis 候选上进行 min-max 归一化；
5. 将方向统一为 `higher_is_better`；
6. 跳过缺失、损坏或无效图片；
7. 从同时满足 `analysis=true` 和 `segmentation_training=true` 的记录中选择最高分的 548 张；
8. 分数相同时按照 `synthetic_id` 升序打破平局，保证结果可复现。

该方法不使用真实图像和 mask，因此属于无参考图像质量指标。SSIM、LPIPS 等全参考指标只有在真实图和合成图几何严格对齐时才能使用，目前不在第一版范围内。

## 3. 数据准备结果

数据存放在 FAU Woody HPC：

```text
/home/woody/srsu/srsu119v/ws-2026-27/datasets/
├── sensation_ds_v2_1_1152x768/
│   ├── training/
│   ├── validation/
│   └── testing/
└── synthetic-data/
    ├── group1/
    ├── group2/
    ├── ...
    └── group9/
```

这里的 `synthetic-data/group1` 到 `group9` 是合成数据的来源分组，不是当前课程小组的编号。当前 Group 2 方法需要处理所有满足条件的来源组，不能只处理 `synthetic-data/group2`。

完整合成数据约 24 GB，共包含 78,883 张图片。

## 4. Canonical manifest

使用共享 manifest 构建程序，在 Slurm 计算节点上生成：

```bash
sbatch group-2/scripts/build_manifest.slurm
```

正式运行结果：

| 项目 | 数量 |
|---|---:|
| manifest 总记录 | 78,883 |
| `analysis=true` | 78,840 |
| `segmentation_training=true` | 43,800 |

输出文件：

```text
synthetic-data-seminar/manifest.jsonl
```

文件约 79 MB。Slurm Job `12841196` 成功完成，用时 21 分 10 秒。

## 5. 全量质量评分与子集选择

正式评分通过以下脚本提交：

```bash
sbatch group-2/scripts/score_quality.slurm
```

也可以直接调用 CLI：

```bash
export PYTHONPATH="$PWD/src:$PWD/group-2/src"

python -m group_2.cli \
  --data-root datasets \
  --manifest synthetic-data-seminar/manifest.jsonl \
  --scores-output group-2/results/fallback-v1/scores.jsonl \
  --subset-output group-2/results/fallback-v1/perceptual_quality_budget_0.25x.jsonl
```

正式运行结果：

| 输出 | 记录数 | 大小 |
|---|---:|---:|
| `scores.jsonl` | 78,840 | 约 55 MB |
| `perceptual_quality_budget_0.25x.jsonl` | 548 | 约 240 KB |

评分任务 Slurm Job `12841210` 用时约 2 小时 4 分钟。

## 6. 输出字段

### 6.1 Score JSONL

每行包括：

- `schema_version`；
- `synthetic_id`；
- `method`；
- `method_version`；
- `direction`；
- `score`；
- `components`；
- `provenance`。

其中 provenance 记录指标名称、灰度预处理、几何处理以及归一化方法。

### 6.2 Subset JSONL

每行包括：

- `synthetic_id`；
- `selected`；
- `rank`；
- `method`；
- `budget`；
- `source_group`；
- `score`；
- `score_method`；
- `score_method_version`；
- `manifest_sha256`。

Subset 只保存记录 ID 和选择信息，不复制原始图片。

## 7. 验证结果

原版本记录了本地和 HPC 的 29 项测试通过。整合后在当前本地环境重新运行 Group 2
及共享 Manifest、评分、选择接口测试，结果为：

```text
32 passed
```

完整结果还进行了额外验证，结果为：

```text
VALIDATION PASSED
manifest records: 78883
scored records: 78840
training candidates with scores: 43800
selected records: 548
score range: 0.0 to 1.0
```

验证内容包括：

- score ID 无重复；
- subset ID 无重复；
- subset 恰好包含 548 条记录；
- rank 严格为 1 到 548；
- 所有选中记录均满足 analysis 和 segmentation training 资格；
- subset 中的分数与 score overlay 一致；
- 548 条记录确实是全部合格候选中的确定性 Top 548。

整合时又使用当前仓库中的正式 Manifest 和结果执行了只读验证：78,883 条
Manifest 记录全部通过 schema `1.1`；78,840 条分数恰好覆盖所有
`analysis=true` 记录；原始分量与 min-max 归一化分数一致；548 条 subset 可由
当前代码确定性重建，并可被公共 `SyntheticSegmentationDataset` 构造器接受。
Manifest SHA-256 也与 HPC 日志及全部 subset 行一致。可复现该检查：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:group-2/src \
  group-2/.venv/bin/python group-2/scripts/validate_results.py
```

当前本地没有 `datasets/`，因此这次整合无法逐图重算 78,840 个原始分数；这不影响
对 JSONL、归一化、资格过滤、Top-548 和公共训练读取接口的验证。

## 8. 当前发现与风险

当前全体 analysis 候选中的 Top 10 分数记录全部来自合成数据来源 `group4`。
`group4` 没有训练合格记录，所以这些图片不会进入最终 548 张 subset。这并不一定
表示实现错误，但可能说明 contrast-gradient fallback 偏好某种高对比度、强锐化
或高频噪声风格。

因此，必须检查最高分与最低分示例图，确认：

- 高分图是否真的更清晰、自然；
- 高分图是否存在明显噪声或过度锐化；
- 低分图是否确实模糊、灰暗或低对比度；
- 分数差异是否主要来自不同生成来源的风格差异。

该现象需要在报告中如实说明。感知质量只是一个研究假设，不等同于下游分割效用。

## 9. 周四提交前剩余工作

- [x] 配置 HPC 和 Python 3.12 环境；
- [x] 下载并解压真实与合成数据；
- [x] 实现第一版无参考质量指标；
- [x] 实现 JSONL 输出和确定性 Top-548 选择；
- [x] 完成单元测试与接口测试；
- [x] 生成 canonical manifest；
- [x] 完成 78,840 张图片的全量评分；
- [x] 生成并验证 548 张 subset；
- [x] 生成并人工检查 Top/Bottom 示例图；
- [x] 统计 548 张图片的 source-group 分布；
- [x] 将正式运行数据和局限补充到英文 README；
- [ ] 由两位组员共同检查代码与结果；
- [ ] 提交到课程 GitLab 的 `main`。

## 10. 最终项目后续工作

以下内容属于完整项目后续阶段，不是第一版 fallback 流程的一部分：

1. 增加第二个无参考指标，例如 BRISQUE 或 NIQE；
2. 对两个指标进行方向和尺度校准；
3. 生成 metric agreement CSV；
4. 分析不同指标之间的一致性；
5. 生成分数分布、Top/Bottom 示例等图表；
6. 使用最终 548 张子集运行 Group 2 方法实验；
7. 使用种子 17、42、73；
8. 与老师提供的 `real_only` 和 `random_budget_0.25x` 比较；
9. 汇报 mean ± standard deviation；
10. 分析图像质量分数与下游分割效用之间的关系。

## 11. 重要限制

- contrast-gradient 可能错误地奖励噪声、锐化和极端对比度；
- min-max 归一化依赖当前候选池；
- 当前只实现了一个无参考指标；
- 当前尚未完成三个种子的下游训练；
- 当前结果只能证明评分和筛选流程可复现，不能证明所选图片一定提升语义分割性能。


