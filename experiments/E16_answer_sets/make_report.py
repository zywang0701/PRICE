"""Write a reproducible result, including negative findings and all families."""
import hashlib
import json
import platform
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import data as d
import model as m
from controller import AnswerSetSelector
from split_pool import bootstrap


def read(path):return json.loads(Path(path).read_text())


def verify_streaming(tag):
    raw=d.load(tag,'test');replay=np.load(d.HERE.parent/'E02_frozen'/tag/'replay.npz')
    expected=np.load(d.HERE/tag/'temperatures.npz');checks=0
    for family in m.FAMILIES:
        for i in [0,17,58]:
            policy=AnswerSetSelector(tag,family,none_cluster=int(raw['none'][i]))
            for path_index in [0,7]:
                policy.reset()
                for n,r in enumerate(replay['perms'][i,path_index,:64]):
                    value=policy.observe(raw['phi'][i,r],raw['clu'][i,r],raw['ell'][i,r])
                    assert value['index']==expected[family][i,path_index,n],(tag,family,i,n)
                    checks+=1
    return checks


def verify_training_labels(tag):
    z=np.load(d.HERE/tag/'train_features.npz');w=z['winners'];offset=z['offsets']
    current=z['y'][offset[:-1,None]+np.maximum(w,0)]&(w>=0)
    expected=z['corr'][:,:,d.CP-1].reshape(-1,13)
    assert np.array_equal(current,expected)
    split=read(d.HERE/tag/'query_splits.json')
    ids=[set(split[k]) for k in ['fit','tune','audit']]
    assert not (ids[0]&ids[1] or ids[0]&ids[2] or ids[1]&ids[2])
    assert len(set.union(*ids))==7425
    grader={}
    for phase in ['train','test']:
        g=read(d.HERE/tag/f'{phase}_grader_timeout_audit.json')
        assert not g['changed']
        grader[phase]=dict(checks=g['checked'],changed=len(g['changed']),unresolved=len(g['unresolved_errors']))
    return dict(all_training_state_correctness_reconstructed=True,disjoint_query_splits=True,
        longer_timeout_grading_audit=grader)


def main():
    cells={tag:read(d.HERE/tag/'results.json') for tag in ['qwen','llama']}
    selections={tag:read(d.HERE/tag/'selection.json') for tag in cells}
    halves={tag:read(d.HERE/tag/'split_pool_summary.json') for tag in cells}
    verification={};lines=['# 答案组比较与跨 rollout 池温度学习：第一轮实现结果','',
        '**已实现并完成两种模型上的三类新方案训练与评估；所有计算使用本机 CPU，没有新增 rollout 或外部推理调用。** '
        '**本轮尚未获得足以更新 deployed 的稳定提升。** 正式 deployed controller、论文及已 stage 图片未改动。', '',
        '## 先回答计算资源', '',
        '本机为 Apple Silicon、10 核 CPU、16 GiB 内存；PyTorch 2.5.1，MPS 可用、CUDA 不可用。'
        '本轮显式使用 CPU、3 个 PyTorch 线程。现有缓存足够完成此轮；GPU 不是当前瓶颈。'
        'Calibration 每题只有 64 条，若要在那里做 64+64 或更大规模的分池，需要新增 rollout。'
        '测试池每题已有 128 条，可以直接补做 64+64 的稳定性诊断。真正新增生成时才需要推理 GPU 或模型服务。', '',
        '## 校准数据上的分池诊断', '',
        '每题现有 64 条 rollout 分成不重叠的 32+32，四次预设划分、双向交换。'
        '每个半池 16 次重排，在 counts 2/4/8/16/32 上平均。'
        '“同池”在 A 选一个温度并在 A 计分；“跨池”在 A 选好后到 B 计分。'
        '这里相对原 calibration 固定温度，使用修正后的评分。区间按 query bootstrap；重排不算独立 query。', '',
        '| 模型 | 同池 hindsight 收益 | 跨池收益 | 跨池收益 95% 区间 |', '|---|---:|---:|---:|']
    for tag,h in halves.items():
        v=h['cross_half_gain'];lines.append(f'| {tag} | {h["same_half_gain"]["mean_pp"]:+.3f} pp | {v["mean_pp"]:+.3f} pp | [{v["lo_pp"]:+.3f}, {v["hi_pp"]:+.3f}] |')
    lines+=['','**小样本估计的逐题最优温度缺乏稳定迁移。** 这提供了同池 oracle 与可部署收益之间差距的一项实证解释。'
        '但这是 calibration、32 条半池的结果，不能直接推算原测试集 64 条 oracle gap 中有多少属于选择偏差，'
        '也不能否定渐近的逐题最优温度。两个半池仍来自同一个已记录池，不是本轮新生成的独立数据。','',
        '在全部模型冻结、测试评估完成后，利用现有测试池再做了 64+64 检查：'
        '相同四次划分、双向交换，在 counts 2/4/8/16/32/64 平均；这不是上面六个停止运行点的指标。'
        '该补充仅解释稳定性，未用于改选或训练模型。','',
        '| 测试池 64+64 | 同池 hindsight 收益 | 跨池收益 | 跨池收益 95% 区间 |','|---|---:|---:|---:|']
    for tag in cells:
        h=read(d.HERE/tag/'test_transfer_summary.json');v=h['cross_half_gain']
        lines.append(f'| {tag} | {h["same_half_gain"]["mean_pp"]:+.3f} pp | {v["mean_pp"]:+.3f} pp | [{v["lo_pp"]:+.3f}, {v["hi_pp"]:+.3f}] |')
    lines+=['',
        '## 实现了什么','',
        '- `answer_mlp`：每个答案组使用票数、分数分位数、前三高分、高分支持、前后半段支持、去掉最高分后的变化，以及 13 个温度下的权重份额。',
        '- `answer_set`：在相同输入上加入全部答案组的平均／最大编码，学习组间比较。两者用多标签正确性和可达答案之间的 pairwise loss 训练；先比较不同答案，再选择支持该答案且离固定 rung 最近的温度。',
        '- `cross_pool`：用 A 半池的已观察前缀预测 B 半池的温度收益曲线；每个前缀共享同一个 B 池目标，不使用唯一最优温度标签。测试时 32 条后冻结估计。',
        '- 前两者只在预测优势超过校准阈值时切换，持续在 1/2/4/8/16/32/48/64 条更新；三个模型均不接收预算或正确标签。', '',
        '每种基础 LLM 使用全部 7,425 道 calibration query，按 query 分成 70%/15%/15% 的拟合、调参、方案选择子集；'
        '无正确 rollout 的题保留。每个 epoch 每题抽两个状态；宽度 48，候选 epoch 15/30/60，阈值 0/.02/.05/.1/.2。'
        '温度网格全部 13 个值均可选，固定比较温度仅在拟合子集按修正评分选定。','',
        '## 冻结选择与测试结果','',
        '测试使用 E15 保存的六个共同停止运行点。所有方法的 rollout 数和 token 成本逐路径相同；'
        '下面是相对本轮 calibration 选定固定温度的准确率变化，不是重新优化各自 frontier 后的比较。'
        '六个实际风险预算与论文标称预算略有差别，完整数值保存在 `results.json`。','']
    for tag,r in cells.items():
        sel=selections[tag];outcomes=np.load(d.HERE/tag/'outcomes.npz');primary=sel['primary']
        selected_delta=outcomes['stops__'+primary].mean((1,2))-outcomes['stops__fixed'].mean((1,2))
        overall=bootstrap(selected_delta)
        s=r['results']['stops'][primary]['versus_fixed']
        lines += [f'### {tag}', '',
            f'Calibration 方案选择：**{primary}**；固定 τ={sel["tau_fixed"]:g}。'
            f'所选方案六个测试运行点的平均变化为 **{overall["mean_pp"]:+.3f} pp**，'
            f'query-bootstrap 95% 区间 [{overall["lo_pp"]:+.3f}, {overall["hi_pp"]:+.3f}]。', '',
            '| 方案 | '+ ' | '.join(f'{x/1000:.1f}k' for x in r['actual_risk_budgets'])+' |',
            '|---|'+'---:|'*6]
        for family in m.FAMILIES+['previous_neural','old_deployed']:
            values=r['results']['stops'][family]['versus_fixed']['mean_pp']
            lines.append('| '+family+' | '+' | '.join(f'{x:+.3f}' for x in values)+' |')
        lines+=['','单位均为 pp。previous_neural 是上一轮 E14 calibration 选出的模型，old_deployed 是原 controller；'
            '此处都重新按本轮统一评分计分。上述方案没有按测试结果重新选择。','',
            '| 新方案 | 独立 calibration 子集变化 | epoch | 切换阈值 |','|---|---:|---:|---:|']
        for name,p in sel['selections'].items():
            lines.append(f'| {name} | {p["audit_delta_pp"]:+.3f} pp | {p["epoch"]} | {p["threshold"]} |')
        lines+=['','之前定位的具体题目，在六个相同停止运行点上的单题平均正确率：','',
            '| query_id | 固定温度 | 之前神经模型 | answer_mlp | answer_set | cross_pool |',
            '|---|---:|---:|---:|---:|---:|']
        for case in r['cases']:
            acc=case['accuracy_pct'];lines.append(f'| {case["query_id"]} | '+' | '.join(f'{acc[x]:.1f}%' for x in ['fixed','previous_neural']+m.FAMILIES)+' |')
        lines+=['']
        stream=verify_streaming(tag)
        checksum=hashlib.sha256((d.HERE/tag/'selection.json').read_bytes()).hexdigest()
        assert checksum==r['selection_sha256']
        assert read(d.HERE/tag/'test_transfer_summary.json')['selection_hash']==checksum
        verification[tag]=dict(exact_streaming_decisions_checked=stream,selection_hash_unchanged=True,
            exact_old_votes_verified=r['exact_old_votes_verified'],same_tokens_verified=r['same_tokens_verified'],
            overall_selected_vs_fixed=overall,post_test_diagnostic_did_not_change_selection=True,
            **verify_training_labels(tag))
    lines += ['## 评分修正范围与仍然存在的限制','',
        '旧代码只把编号最小的正确 cluster 作为目标。新实验保留原答案分组，对有冲突的 query，'
        '将每个 cluster 的规范代表答案与 gold 重新评分，并接受所有判对的 cluster。'
        '不会利用 gold 合并投票组。修正的是监督和计分；没有全面重做答案等价分组，也没有人工认证全部 gold/grader。', '',
        '| 模型 | Calibration 冲突题 | 测试冲突题 | 多正确 cluster 的 calibration 题 |','|---|---:|---:|---:|']
    for tag in cells:
        a=read(d.HERE/tag/'train_scoring_audit.json');b=read(d.HERE/tag/'test_scoring_audit.json')
        lines.append(f'| {tag} | {a["conflicted_queries"]} | {b["conflicted_queries"]} | {a["queries_with_multiple_correct_clusters"]} |')
    lines+=['','与 E15 的 calibration 冲突数不同，本轮检查全部 7,425 道题，包含旧学习缓存排除的题，'
        '并把 abstention 与原始正确标记的冲突也记入。评分使用 installed math-verify 加规范字符串回退；'
        '额外保存 12 位小数舍入敏感性结果和排除原冲突题的测试结果。这个严格数值视图只重评冲突题，'
        '不是全数据严格评分重跑。Qwen metadata 对齐抽查的两次失败均为正确标记对应的答案已被规范化成 `<none>`；'
        '这属于规范化与评分间的冲突，不能直接归咎于 grader 或 query ID 错位。本轮没有恢复这些丢失的文本答案。', '',
        '本轮模型仅使用统计证据，没有加入 reasoning 文本 verifier；有限池和 scorer 的信息限制仍在。'
        'MATH-500 已用于此前诊断，所以本轮属于探索性评估；bootstrap 区间条件于已拟合模型，'
        '不包含重训、多候选选择或重复使用测试集的不确定性。', '',
        '## 使用与复现','',
        '可逐条调用 `controller.AnswerSetSelector(tag).observe(score, answer_cluster, token_count)`，'
        '返回温度索引和值；score 须使用 calibration CDF 变换。调用者仍负责答案分组和原停止规则。'
        '该接口是实验候选，尚未替换正式 controller。', '',
        '在实验目录使用 `python`，对 `--cell qwen` 和 `--cell llama` 分别运行：', '',
        '```text','data.py --phase train','features.py','split_pool.py','model.py','evaluate.py',
        'audit_grader.py --phase train','audit_grader.py --phase test','test_transfer.py','```','',
        '每行均补上 `--cell` 参数；`evaluate.py` 自动准备测试缓存。最后运行 '
        '`python -m unittest -v test_invariants` 和 `python make_report.py`。'
        '复现顺序保证模型选择先于新的测试评估；生成过程不会改写原实验和论文。', '',
        f'- [完整协议]({d.HERE}/PROTOCOL.md)',f'- [逐条调用接口]({d.HERE}/controller.py)',
        f'- [Qwen 完整结果与置信区间]({d.HERE}/qwen/results.json)',
        f'- [Llama 完整结果与置信区间]({d.HERE}/llama/results.json)',
        f'- [Qwen 分池诊断]({d.HERE}/qwen/split_pool_summary.json)',
        f'- [Llama 分池诊断]({d.HERE}/llama/split_pool_summary.json)',
        f'- [核对结果]({d.HERE}/verification.json)', '']
    fig,axes=plt.subplots(1,3,figsize=(14,4.2))
    x=np.arange(2);width=.34
    for j,(key,label,color) in enumerate([('same_half_gain','Choose and score on A','#989898'),('cross_half_gain','Choose on A, score on B','#2b6f9b')]):
        vals=np.array([halves[t][key]['mean_pp'] for t in cells]);low=np.array([halves[t][key]['lo_pp'] for t in cells]);hi=np.array([halves[t][key]['hi_pp'] for t in cells])
        axes[0].bar(x+(j-.5)*width,vals,width,color=color,label=label,yerr=np.stack([vals-low,hi-vals]),capsize=3)
    axes[0].set_xticks(x,['Qwen','Llama']);axes[0].set_title('Calibration: 32+32 rollout split');axes[0].legend(fontsize=8)
    for ax,(tag,r) in zip(axes[1:],cells.items()):
        for family in m.FAMILIES:
            ax.plot(np.arange(6),r['results']['stops'][family]['versus_fixed']['mean_pp'],'o-',markersize=3,label=family)
        ax.set_xticks(np.arange(6),[f'{v/1000:.1f}k' for v in r['actual_risk_budgets']],rotation=35)
        ax.set_title(f'{tag.capitalize()}: same stopping paths');ax.set_xlabel('Actual entropic-risk budget (tokens)');ax.legend(fontsize=8)
    for ax in axes:ax.axhline(0,color='black',linewidth=.6);ax.set_ylabel('Accuracy gain (pp)');ax.grid(axis='y',alpha=.15)
    fig.tight_layout();fig.savefig(d.HERE/'answer_set_results.png',dpi=180);fig.savefig(d.HERE/'answer_set_results.pdf')
    lines[4:4]=[f'![Split-pool and frozen-model results]({d.HERE}/answer_set_results.png)','']
    text='\n'.join(lines)+'\n';(d.HERE/'REPORT.md').write_text(text)
    paper=Path(d.C.PAPER)/'answer_set_voting_experiments.md';paper.write_text(text)
    verification['environment']=dict(python=sys.version,platform=platform.platform(),torch=torch.__version__,
        cuda=torch.cuda.is_available(),mps=torch.backends.mps.is_available(),used_device='cpu')
    verification['model_parameters']={family:sum(p.numel() for p in m.AnswerNet(family).parameters()) for family in m.FAMILIES}
    d.dump(d.HERE/'verification.json',verification)
    files=[p for p in d.HERE.rglob('*') if p.is_file() and '__pycache__' not in str(p) and p.name!='artifact_manifest.json' and p.suffix!='.log']+[paper]
    d.dump(d.HERE/'artifact_manifest.json',{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)})
    d.log('Wrote',paper)


if __name__=='__main__':main()
