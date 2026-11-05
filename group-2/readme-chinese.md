# Group 2：感知图像质量筛选

## 研究问题

在合成图片预算固定为 **548 张**时，按照感知质量选择合成图片，是否能够提高真实
道路场景的语义分割性能？

下游实验计划使用 DeepLabV3+，种子为17、42、73，并与 `real_only` 和
`random_budget_0.25x` 比较。目前尚未运行下游训练。

## 方法和结果

| 方法 | 说明 | 当前结果 |
| --- | --- | --- |
| 1. Contrast-gradient fallback | 灰度标准差加水平、垂直绝对梯度；越高越好 | **已完成：** 78,840条分数、0 skipped、确定性Top-548 |
| 2. NIQE | pyiqa无参考自然场景统计指标；原始分数越低越好 | 代码和真实HPC Smoke Test已完成；**尚无全量结果** |
| 3. BRISQUE | pyiqa无参考空间自然场景统计指标；原始分数越低越好 | 代码和真实HPC Smoke Test已完成；**尚无全量结果** |

已经完成正式结果的 fallback 公式为：

```python
gray = np.mean(rgb, axis=2)
raw_score = (
    np.std(gray)
    + np.mean(np.abs(np.diff(gray, axis=0)))
    + np.mean(np.abs(np.diff(gray, axis=1)))
)
```

分数通过 min-max 归一化到 `[0,1]`。筛选时只保留同时满足
`eligibility.analysis=true` 和 `eligibility.segmentation_training=true` 的记录，
按 `(-score, synthetic_id)` 排序，固定选择548张。

NIQE 和 BRISQUE 使用 `pyiqa==0.1.16`。先保存原始分数，只有完整评分池处理完毕后
才统一转换为越高越好的归一化分数。两个真实HPC Smoke Test均完成8/8张图片、
0 skipped。它们目前没有全量结果，因为当前逐张安全评分流程较慢，处理78,840张
预计需要**数十小时**，可能需要多次HPC断点恢复。Smoke输出不作为正式研究结果。

## 已有的 fallback 正式结果

| 输出 | 结果 |
| --- | --- |
| `results/fallback-v1/scores.jsonl` | 78,840条归一化分数 |
| `results/fallback-v1/perceptual_quality_budget_0.25x.jsonl` | 548个唯一入选ID |
| `results/fallback-v1/top_bottom_examples.jpg` | 全局Top-10/Bottom-10可视化 |
| `results/fallback-v1/source_group_distribution.csv` | group3: 309；group5: 141；group7: 69；group9: 29 |
| `results/fallback-v1/source_group_eligibility_and_selection.csv` | group1–group9资格与入选率 |

![Fallback Top-10和Bottom-10](results/fallback-v1/top_bottom_examples.jpg)

以上是质量评分与筛选结果，不是语义分割性能结果。

## 项目结构

### 文件夹

| 文件夹 | 用途 |
| --- | --- |
| `src/group_2/` | 评分与筛选代码 |
| `scripts/` | 验证与分析入口 |
| `scripts/hpc/` | NIQE/BRISQUE HPC运行 |
| `tests/` | 自动化测试 |
| `results/fallback-v1/` | 已完成的fallback正式结果 |
| `logs/` | 历史Slurm日志 |

### 文件

| 文件 | 用途 |
| --- | --- |
| `README.md` | 英文项目说明 |
| `readme-chinese.md` | 中文项目说明 |
| `HPC_NIQE_BRISQUE.md` | HPC运行说明 |
| `requirements-iqa.txt` | NIQE/BRISQUE依赖 |
| `.gitignore` | 本地IQA环境忽略规则 |
| `src/group_2/perceptual_quality.py` | fallback评分与Top-548 |
| `src/group_2/cli.py` | fallback命令行入口 |
| `src/group_2/nr_iqa.py` | NIQE/BRISQUE评分管线 |
| `src/group_2/nr_iqa_cli.py` | NIQE/BRISQUE命令行入口 |
| `src/group_2/metric_comparison.py` | 指标比较统计与绘图 |
| `scripts/build_manifest.slurm` | Manifest生成任务 |
| `scripts/score_quality.slurm` | fallback评分任务 |
| `scripts/make_quality_report.py` | fallback图和CSV报告 |
| `scripts/validate_results.py` | 正式结果验证 |
| `scripts/compare_metrics.py` | 三指标比较入口 |
| `scripts/hpc/run_nr_iqa.sh` | HPC共用运行脚本 |
| `scripts/hpc/score_niqe.slurm` | NIQE Slurm模板 |
| `scripts/hpc/score_brisque.slurm` | BRISQUE Slurm模板 |
| `tests/test_group_2.py` | fallback与公共接口测试 |
| `tests/test_nr_iqa.py` | 新指标与比较分析测试 |
| `results/README.md` | 结果状态说明 |
| `results/fallback-v1/scores.jsonl` | fallback正式分数 |
| `results/fallback-v1/perceptual_quality_budget_0.25x.jsonl` | fallback正式Top-548 |
| `results/fallback-v1/top_bottom_examples.jpg` | fallback可视化 |
| `results/fallback-v1/source_group_distribution.csv` | 入选来源组分布 |
| `results/fallback-v1/source_group_eligibility_and_selection.csv` | 各来源组入选统计 |
| `logs/build-manifest-12841194.out` | 第一次Manifest失败日志 |
| `logs/build-manifest-12841196.out` | Manifest成功日志 |
| `logs/score-quality-12841210.out` | fallback成功日志 |

## 验证结果

- 本地自动化测试：**39 passed，2 skipped**。
- NIQE HPC Smoke：**8 valid，0 skipped**。
- BRISQUE HPC Smoke：**8 valid，0 skipped**。
- fallback正式验证：**78,840条分数，恰好548条入选记录**。

## 下一步工作

1. 断点继续NIQE和BRISQUE，完成全部78,840张评分。
2. 生成两个指标的归一化分数、Top-548和比较图。
3. 比较相关性、Top-548重合率和来源组入选率。
4. 使用种子17、42、73训练DeepLabV3+。
5. 与 `real_only`、`random_budget_0.25x` 在真实验证集和测试集上比较。

在下游实验完成前，质量分数更高或不同质量指标相关，不能被解释为语义分割性能提高。
