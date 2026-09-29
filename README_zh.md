<div align="center">

# PRICE

### 大语言模型测试时计算的帕累托前沿：<br>Token 预算下的自适应采样与投票

[Zhenyu Wang](https://zywang0701.github.io/)<sup>1</sup> · Xiaozhi Zhu<sup>2</sup> · [Yifan Hu](https://sites.google.com/view/yifan-hu)<sup>1</sup>

<sup>1</sup> Rutgers University &nbsp; <sup>2</sup> Meta

<p>
  <a href="https://huggingface.co/datasets/zach-wang/PRICE-rollouts"><img src="assets/badge-data.svg" alt="下载采样数据" height="30"></a>
  <a href="#quick-start"><img src="assets/badge-start.svg" alt="快速开始" height="30"></a>
  <a href="#results"><img src="assets/badge-results.svg" alt="查看实验结果" height="30"></a>
</p>

[English](README.md) · **中文**

[简介](#introduction) · [理论](#theory) · [实验结果](#results) · [快速开始](#quick-start) · [实验复现](experiments/README.md) · [引用](#citation)

</div>

<a id="introduction"></a>

## 简介

增加采样次数可以提高大语言模型回答的准确率，但每次生成都会消耗 tokens。当一批问题共享有限预算时，需要同时决定：**每道题生成多少次，以及如何对生成的答案投票。**

**PRICE**（**P**riced **R**ollouts and **I**nference-time voting-rule **C**hoice under a token budg**E**t）为每道题联合选择采样次数与投票规则，以提高给定 token 预算下的准确率。它用一个共享价格权衡预期准确率增益与继续生成的成本：容易的问题提前结束，值得投入计算的问题获得更多采样。

投票规则涵盖自一致性多数投票、基于分数的加权投票与 best-of-*n*。PRICE 根据问题选择规则，并在预测收益不再覆盖定价后的成本时停止。

<p align="center">
  <a href="assets/fig_illustration.png"><img src="assets/fig_illustration.png" width="100%" alt="PRICE 为不同问题选择采样次数和投票规则；示意图展示自适应策略优于固定投票规则的成本–准确率前沿。"></a>
  <br><sub><b>图 1.</b> 联合决定生成多少次、如何投票。右侧前沿为示意图。</sub>
</p>

<a id="theory"></a>

## 理论

- **自适应投票提高准确率上限。** 不同投票规则能够解决的问题不同；逐题选择规则可以覆盖这些可解集合的并集，因此准确率上限不低于任一固定规则。
- **不存在对所有问题分布和预算都最优的固定规则。** 固定规则的前沿可能相交，而自适应前沿在每个预算下均不低于任一固定规则的前沿。
- **大预算下，自适应前沿具有闭式刻画。** 准确率以指数速度接近上限，速率由仍然可解的最难问题决定；实验中的速率验证支持这一规律。

详见[理论、配套图示与算法说明](docs/method.md)（英文）。

<a id="results"></a>

## 实验结果

在 **MATH-500** 上，使用 **Qwen2.5-1.5B** 与 **Llama-3.2-3B**：

- **PRICE-oracle** 使用带标签的采样池，衡量联合自适应的潜在收益。在相同预算下，自适应采样次数带来 **2–9 个准确率百分点**的提升；在采用自适应次数、事后选出的最优固定投票规则基础上，再逐题调整投票规则可额外提升 **2–3 个百分点**。
- **PRICE-deployed** 使用校准数据训练的预测器，以及测试时已生成样本的无标签统计量。在所有报告的预算下，均优于参与比较的六种可部署基线，且预算越紧，优势越明显。

| PRICE-deployed | Qwen2.5-1.5B | Llama-3.2-3B |
| --- | --- | --- |
| 最紧预算下，相对最佳基线的准确率提升 | **+6.4 个百分点** | **+7.3 个百分点** |
| 同等准确率下，相对自一致性投票的最大 token 节省倍数 | **2.8×** | **3.5×** |

<p align="center">
  <a href="assets/fig_ptrue_deployed_frontier.png"><img src="assets/fig_ptrue_deployed_frontier.png" width="100%" alt="Qwen2.5-1.5B 与 Llama-3.2-3B 在 MATH-500 上的成本–准确率前沿：PRICE-deployed 与自一致性投票、ESC 的比较。"></a>
  <br><sub><b>图 2.</b> PRICE-deployed 在相同预算下取得更高准确率，在同等准确率下使用更少 tokens。</sub>
</p>

准确率增益以百分点计。预算采用论文中的**风险调整 token 成本**，同时考虑高成本尾部。详见[完整结果与评估协议](docs/results.md)和[预算定义](docs/method.md#the-token-budget)（英文）。

<a id="quick-start"></a>

## 快速开始

### 1. 运行算法示例——无需 GPU 或数据集

使用 **Python 3.10+**，从新克隆的仓库开始：

```bash
git clone https://github.com/zywang0701/PRICE.git
cd PRICE
python -m venv .venv
source .venv/bin/activate
python -m pip install "numpy>=1.26"
python -m price.demo
```

示例包含三道合成问题：很快就能确定答案的问题、值得增加计算的问题，以及继续采样不再划算的问题。它会输出采样次数、投票规则、预算–准确率曲线与顺序停止决策。示例中的曲线由带标签的合成池估计；论文中的部署预测器另行使用 MATH-train 训练。

### 2. 下载已发布的采样池

论文实验在 CPU 上回放已生成的样本。四个“模型–数据集”组合的数据共约 **0.8 GB**；下载脚本会根据数据清单核验文件哈希。

```bash
python -m pip install -r requirements.txt
python scripts/download_data.py
```

### 3. 复现论文结果

按以下顺序运行；部署实验依赖 oracle 分支生成的中间产物：

```bash
bash experiments/run_all.sh oracle       # PRICE-oracle 与共享回放产物
bash experiments/run_all.sh deployed     # PRICE-deployed 与六种基线
bash experiments/run_all.sh paper        # 将表格与图输出到 build/
```

Oracle 分支已使用发布数据重新执行，并精确复现论文对应结果。部署分支按论文运行时的代码发布，打包后未重新完整执行。完整依赖顺序与发布范围见[复现指南](experiments/README.md)（英文）。

<details>
<summary><b>测试、输出位置与可选 GPU 阶段</b></summary>

```bash
python -m pytest tests/
```

下载的数据保存在 `outputs/cells/`，论文表格与图分别输出到 `build/tables/` 和 `build/images/`。自定义位置可使用 `PRICE_ROOT` 与 `PRICE_PAPER_DIR`，详见[环境设置](experiments/README.md#setup)。

如需生成并评分新的采样池，使用 [`generation/`](generation/) 下的脚本，并在 CPU 依赖之外安装 [`requirements-gpu.txt`](requirements-gpu.txt)。SLURM 脚本中需要适配的集群设置标记为 `## EDIT`，详见[生成与评分说明](experiments/README.md#generation-and-scoring-optional-gpu)。

</details>

## 代码导览

[`price/`](price/) 提供两个算法的简洁 NumPy 实现。[`experiments/`](experiments/) 包含生成论文数值所用的校准、回放与评估流程。

| 模块 | 实现 |
| --- | --- |
| Boltzmann 加权投票 | [`price/vote.py`](price/vote.py) |
| Oracle 曲线、定价决策与预算搜索 | [`price/oracle.py`](price/oracle.py) |
| 顺序停止、成本估计与在线价格更新 | [`price/deployed.py`](price/deployed.py) |
| 合成示例 | [`price/demo.py`](price/demo.py) |
| 论文实验链 | [`experiments/run_all.sh`](experiments/run_all.sh) |

[方法说明](docs/method.md) · [完整结果](docs/results.md) · [实验复现](experiments/README.md) · [采样数据](https://huggingface.co/datasets/zach-wang/PRICE-rollouts)

<a id="citation"></a>

## 引用

```bibtex
@misc{wang2026price,
  title  = {Pareto Frontier of LLM Test-Time Compute: Adaptive Rollouts and Voting Under Token Budgets},
  author = {Wang, Zhenyu and Zhu, Xiaozhi and Hu, Yifan},
  year   = {2026},
  note   = {Preprint}
}
```

## 许可

代码采用 [MIT License](LICENSE)。采样数据采用 **CC BY 4.0**，同时遵循 [DATA_LICENSE.md](DATA_LICENSE.md) 中列出的上游模型与数据集条款。
