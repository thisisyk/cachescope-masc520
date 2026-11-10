"""Build a readable summary solely from completed, measured experiment files."""
from pathlib import Path
import csv
import html
import json

ROOT=Path(__file__).resolve().parent

def pct(value):
    return f"{100*value:.2f}%"

def main():
    report=json.loads((ROOT/'results/metrics.json').read_text(encoding='utf8'))
    training=json.loads((ROOT/'models/training_report.json').read_text(encoding='utf8'))
    pilot=json.loads((ROOT/'results/pilot_labeling_summary.json').read_text(encoding='utf8'))
    metadata=json.loads((ROOT/'results/answering_model.json').read_text(encoding='utf8'))
    length_path=ROOT/'results/length_stress_metrics.json'
    length_report=json.loads(length_path.read_text(encoding='utf8')) if length_path.exists() else None
    overhead_path=ROOT/'results/decision_overhead.json'
    variability_path=ROOT/'results/output_variability.json'
    overhead=json.loads(overhead_path.read_text()) if overhead_path.exists() else None
    variability=json.loads(variability_path.read_text()) if variability_path.exists() else None
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure_folder=ROOT/'results/figures'
    figure_folder.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(10.5,4),constrained_layout=True)
    for seed in [520,521,522]:
        history=json.loads((ROOT/f'models/mlp_seed_{seed}_history.json').read_text())
        axes[0].plot([r['epoch'] for r in history],[r['train_loss'] for r in history],label=f'Seed {seed}')
        axes[1].plot([r['epoch'] for r in history],[r['validation_loss'] for r in history],label=f'Seed {seed}')
    for axis,title in zip(axes,['Training loss','Validation loss (early stopping)']):
        axis.set(xlabel='Epoch',ylabel='Weighted binary cross-entropy',title=title)
        axis.grid(alpha=.2)
        axis.legend()
    fig.savefig(figure_folder/'mlp_training_curves.png',dpi=170)
    plt.close(fig)
    policies=['original','always_60','always_30','simple_rule_gate','lr','rf','mlp']
    names={'original':'原文','always_60':'固定保留60%','always_30':'固定保留30%',
           'simple_rule_gate':'简单规则门控','lr':'Logistic Regression 门控','rf':'Random Forest 门控','mlp':'MLP 门控（预定主模型）'}
    lines=['# CacheScope 实验结果','',
      '已完成真实本地回答模型标注、风险模型训练和任务级策略比较。所有任务均为受控合成数据。','',
      '主要发现：主测试集的MLP输入token减少32.60%，准确率由76%变为79%，但准确率变化的95%分组区间包含零。独立结构留出集的准确率由86%降至68%，表明当前风险模型存在明显的跨结构泛化问题。','',
      f"回答模型：`{metadata['model']}`；模型指纹：`{metadata['model_digest']}`。",
      f"开发集：{pilot['tasks']} 个任务、{pilot['candidate_records']} 条候选，原文准确率 {pct(pilot['original_accuracy'])}，观察到 {pilot['observed_regressions']} 个退化候选。",'',
      '正式集 500 个任务（训练300／验证100／测试100），另有50个合成结构留出任务。每任务有两档候选。开发集和结构留出集不参加训练。',
      '模型仅在原文回答正确的训练候选上学习条件风险；最终策略在全部测试任务上评估。',
      '原文、两份候选均由同一个实际模型回答，标签由预定义事实的严格归一化精确匹配生成，没有规则模拟的回答标签。','']
    for split,title in [('test','正式测试集'),('ood','独立结构留出集')]:
        lines += [f'## {title}','',
          '| 策略 | 准确率 | 相对原文变化 | 完整输入token减少 | 采用压缩 | 新增错误数 |',
          '|---|---:|---:|---:|---:|---:|']
        for name in policies:
            m=report['splits'][split][name]
            lines.append(f"| {names[name]} | {pct(m['accuracy'])} | {100*m['accuracy_delta']:+.2f} 个百分点 | {pct(m['input_token_reduction_ratio'])} | {m['accepted_count']}/{m['n_tasks']} | {m['harmful_accepted_count']} |")
        lines += ['']
    m=report['splits']['test']['mlp']
    ci=m['paired_group_bootstrap']
    low,high=ci['accuracy_delta_95_ci']
    sl,sh=ci['input_token_reduction_ratio_95_ci']
    lines += ['## 主模型解释','',
      f"主模型固定为 seed 520 的两层隐藏层 MLP，没有根据测试成绩挑选种子。测试准确率相对原文变化为 {100*m['accuracy_delta']:+.2f} 个百分点；按模板家族重采样的95%区间为 [{100*low:+.2f}, {100*high:+.2f}] 个百分点。",
      f"测试完整输入token减少 {pct(m['input_token_reduction_ratio'])}；对应95%区间 [{pct(sl)}, {pct(sh)}]。",
      f"阈值为 {report['selected_thresholds']['mlp']['threshold']:.6f}，只由验证集选择。验证集上2个百分点的约束是经验选点条件，不是测试集或实际使用的风险保证。",'',
      'MLP、LR、RF及三个MLP种子的全部结果均保留，不能据此预设神经网络一定优于简单模型。',
      '## 结果范围与局限','',
      '- 主数据上下文实测187–247 tokens，属于短文本；额外约1,000-token的无关附录压力测试也不等于真实长文档或业务数据验证。',
      '- 主划分隔离50个措辞家族，但五种高层结构仍在训练和测试中共享；结构留出集只有三种新格式。',
      '- 只有一个0.5B量化回答模型，不能外推到其他模型。',
      '- 分组bootstrap与训练随机种子波动分别报告；测试100任务、10组，结构留出50任务、3组，估计存在不确定性。',
      '- token减少不等于费用或订阅额度减少。调用耗时受缓存和固定顺序影响，不是因果提速证据。',
      '- 原始回答已错误的任务不作为安全训练负例，但始终计入完整策略评测。','',
      '## 查看与复现','',
      '- `results/policy_results.csv`：全部策略数字。',
      '- `results/metrics.json`：阈值、指标和置信区间。',
      '- `results/figures/`：策略对比、权衡曲线和风险PR曲线。',
      '- `results/failures.jsonl`：问题、参考答案、模型回答和失败上下文。',
      '- `data/raw_responses.jsonl`：真实回答、时间、模型指纹和用量。',
      '- `models/`：MLP、LR、RF、两个消融模型及训练记录。',
      '- 安装依赖后运行 `python verify_results.py` 可重新核对标签、特征与结果完整性；`python run_experiment.py train` 可用保存的真实回答重新训练和评测，无需再次调用回答模型。',
      '- 本机现有环境可运行 `powershell -ExecutionPolicy Bypass -File .\\run_local.ps1 -Stage verify`。',
      '- 源码和合成任务由AI辅助编写；团队成员应核查、理解并按课程要求披露实际贡献和AI使用。','']
    if overhead:
        lines += ['## 决策开销实测','',f"MLP 完整本地决策的中位耗时为 {overhead['results']['mlp']['median_ms']:.2f} ms，p95 为 {overhead['results']['mlp']['p95_ms']:.2f} ms；包含两档压缩、分词、特征、预测与选择，不含模型加载和回答调用。详见 `results/decision_overhead.json`。",'']
    if variability:
        lines += ['## 固定设置重复检查','',f"预先指定的20个开发任务共进行 {variability['n_answer_comparisons']} 次回答对照；原始文本改变 {variability['answer_text_changes']} 次，归一化答案改变 {variability['normalized_answer_changes']} 次，正确性改变 {variability['correctness_changes']} 次。重复结果不用于训练或阈值选择。详见 `results/output_variability.json`。",'']
    if length_report:
        lines += ['## 补充长度压力测试','',
          '固定选择50个既有主测试来源，分别添加前置或后置无关附录；上下文1,000–1,013 tokens。复用已训练模型及原验证阈值，不重新调参。这些是配对变体，不是新独立来源，也不是自然长文数据。','',
          '| 策略 | 准确率 | 相对长输入原文变化 | 完整输入token减少 | 新增错误数 |',
          '|---|---:|---:|---:|---:|']
        for name in policies:
            v=length_report['policies'][name]
            lines.append(f"| {names[name]} | {pct(v['accuracy'])} | {100*v['accuracy_delta']:+.2f} 个百分点 | {pct(v['input_token_reduction_ratio'])} | {v['harmful_accepted_count']} |")
        lines += ['', '全部结果见 `results/length_stress_metrics.json` 和 `results/length_stress_policy_results.csv`。','']
    (ROOT/'RESULTS.md').write_text('\n'.join(lines),encoding='utf8')
    tables=[]
    for split,title in [('test','正式测试集（100任务）'),('ood','独立结构留出集（50任务）')]:
        rows=[]
        for name in policies:
            v=report['splits'][split][name]
            rows.append('<tr>'+''.join(f'<td>{html.escape(str(x))}</td>' for x in [names[name],pct(v['accuracy']),f"{100*v['accuracy_delta']:+.2f}pp",pct(v['input_token_reduction_ratio']),v['harmful_accepted_count']])+'</tr>')
        tables.append(f'<h2>{title}</h2><table><thead><tr><th>策略</th><th>准确率</th><th>变化</th><th>输入token减少</th><th>新增错误数</th></tr></thead><tbody>'+''.join(rows)+'</tbody></table>')
    if length_report:
        rows=[]
        for name in policies:
            v=length_report['policies'][name]
            rows.append('<tr>'+''.join(f'<td>{html.escape(str(x))}</td>' for x in [names[name],pct(v['accuracy']),f"{100*v['accuracy_delta']:+.2f}pp",pct(v['input_token_reduction_ratio']),v['harmful_accepted_count']])+'</tr>')
        tables.append('<h2>约1,000-token长度压力测试（50个配对变体）</h2><p>沿用原模型和阈值；不是新增独立来源，也不是自然长文。</p><table><thead><tr><th>策略</th><th>准确率</th><th>变化</th><th>输入token减少</th><th>新增错误数</th></tr></thead><tbody>'+''.join(rows)+'</tbody></table>')
    figures=''.join(f'<img alt="{html.escape(p.stem)}" src="results/figures/{p.name}">' for p in sorted((ROOT/'results/figures').glob('*.png')))
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>CacheScope 实验结果</title><style>body{max-width:1100px;margin:40px auto;padding:0 24px;font:16px/1.7 system-ui,"Microsoft YaHei",sans-serif;color:#17212d;background:#fcfcfa}h1{font-size:30px}h2{margin-top:38px}table{border-collapse:collapse;width:100%;font-size:15px}td,th{padding:10px 12px;border:1px solid #d3d9de;text-align:left}th{background:#e7edf1}img{display:block;width:100%;height:auto;margin:30px 0}a{color:#155b80}.note{color:#526170}</style><h1>CacheScope 实验结果</h1><p>100个开发任务、500个正式任务和50个合成结构留出任务。真实本地Qwen回答生成标签，训练MLP、Logistic Regression和Random Forest，并用验证集锁定门控阈值。</p>'''+''.join(tables)+'''<p class="note">预定主模型为seed 520的MLP。所有数据是受控合成事实记录；本次结果不能证明长文本、其他模型或真实业务场景的通用收益。输入token减少不代表费用或订阅额度减少。</p>'''+figures+'''<p><a href="RESULTS.md">完整结果说明</a> · <a href="results/policy_results.csv">全部策略CSV</a> · <a href="README.md">运行说明</a></p></html>'''
    (ROOT/'实验结果.html').write_text(page,encoding='utf8')
    print('Wrote RESULTS.md and 实验结果.html from measured results.')

if __name__=='__main__':main()
