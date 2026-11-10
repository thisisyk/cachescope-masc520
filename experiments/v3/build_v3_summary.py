"""Render locked V3 metrics; never train, infer, calibrate, or read candidates.

Only results/metrics.json, frozen_selection.json and an optional verification
JSON supply outcomes. No report is created if the measured results are absent.
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LABELS = {
    'original': '原文', 'always_60': '固定保留60%上下文', 'always_30': '固定保留30%上下文',
    'simple_rule_gate': '简单保留率规则', 'guard_only': '仅JSON结构检查',
    'mlp_seed_520': '新训纯MLP（seed 520）', 'mlp_seed_521': '新训纯MLP（seed 521）',
    'mlp_seed_522': '新训纯MLP（seed 522）', 'mlp_ensemble': '新训纯MLP集成（3分数取最大值）',
    'hybrid_mlp_ensemble': '新训MLP集成＋JSON检查',
    'lr': '新训Logistic Regression', 'rf': '新训Random Forest',
    'hybrid_lr': '新训LR＋JSON检查', 'hybrid_rf': '新训RF＋JSON检查',
    'frozen_v2_mlp': '冻结V2 MLP（原门槛）',
    'hybrid_frozen_v2_mlp': '冻结V2 MLP＋JSON检查（新校准）',
}
PLOT_LABELS = {
    'original': 'Original', 'always_60': 'Fixed keep 60%', 'always_30': 'Fixed keep 30%',
    'simple_rule_gate': 'Retention rule', 'guard_only': 'JSON guard only',
    'mlp_ensemble': 'New pure ensemble', 'hybrid_mlp_ensemble': 'New ensemble + guard',
    'frozen_v2_mlp': 'Frozen V2', 'hybrid_frozen_v2_mlp': 'Frozen V2 + guard',
    'lr': 'New LR', 'rf': 'New RF', 'hybrid_lr': 'New LR + guard', 'hybrid_rf': 'New RF + guard',
    'mlp_seed_520': 'Pure MLP 520', 'mlp_seed_521': 'Pure MLP 521', 'mlp_seed_522': 'Pure MLP 522',
}
ORDER = ['original', 'always_60', 'always_30', 'simple_rule_gate', 'guard_only',
         'frozen_v2_mlp', 'hybrid_frozen_v2_mlp', 'mlp_seed_520', 'mlp_seed_521',
         'mlp_seed_522', 'mlp_ensemble', 'hybrid_mlp_ensemble', 'lr', 'rf', 'hybrid_lr', 'hybrid_rf']


def label(name):
    return LABELS.get(name, name)


def plot_label(name):
    return PLOT_LABELS.get(name, name.replace('_', ' '))


def pct(value):
    return '—' if value is None else f'{100 * value:.2f}%'


def pp(value):
    return '—' if value is None else f'{100 * value:+.2f}个百分点'


def number(value):
    return '—' if value is None else f'{value:.3f}'


def pvalue(value):
    if value is None:
        return '—'
    return '<0.0001' if value < .0001 else f'{value:.4f}'


def interval(values, delta=False):
    if values is None:
        return '未提供'
    if delta:
        return f'[{100 * values[0]:+.2f}, {100 * values[1]:+.2f}] 个百分点'
    return f'[{100 * values[0]:.2f}%, {100 * values[1]:.2f}%]'


def ordered_policies(policies):
    # Preserve every actual policy, including policies added after authoring.
    return [n for n in ORDER if n in policies] + sorted(set(policies) - set(ORDER))


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def table_policies(metrics, primary):
    policies = metrics['policies']
    headers = ['策略', '准确率', '输入token节省', '采用压缩', '原文正确→答错', '原文错误→答对']
    rows = []
    for name in ordered_policies(policies):
        m = policies[name]
        rows.append([label(name) + ('【校准入选主方案】' if name == primary else ''),
                     pct(m['accuracy']), pct(m['input_token_reduction_ratio']), m['accepted_count'],
                     m['harmful_accepted_count'], m['improvement_count']])
    return headers, rows


def validate_inputs(metrics, frozen):
    primary = metrics['primary_policy']
    if primary != frozen['primary_policy']:
        raise ValueError('Report primary differs from frozen calibration selection')
    if primary not in ('hybrid_mlp_ensemble', 'hybrid_frozen_v2_mlp'):
        raise ValueError('Unknown predeclared primary hybrid')
    if not metrics.get('primary_is_hybrid') or not frozen.get('primary_is_hybrid'):
        raise ValueError('Primary must be disclosed as a hybrid')
    if metrics['test_tasks'] != frozen['test_expected_tasks']:
        raise ValueError('Measured test count differs from frozen selection')
    if primary not in metrics['policies']:
        raise ValueError('Missing primary policy measurements')
    if metrics['all_predeclared_primary_criteria_met'] != all(metrics['primary_success_criteria_observed'].values()):
        raise ValueError('Overall target status disagrees with individual checks')
    return primary


def make_figures(metrics, frozen, target):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    target.mkdir(parents=True, exist_ok=True)
    primary = metrics['primary_policy']
    policies = metrics['policies']
    preferred = ['original', 'always_60', 'always_30', 'guard_only', 'frozen_v2_mlp',
                 'mlp_ensemble', 'hybrid_mlp_ensemble', 'hybrid_frozen_v2_mlp']
    names = [n for n in preferred if n in policies]
    if primary not in names:
        names.append(primary)
    colors = ['#137a68' if n == primary else '#b9c3cb' if n == 'original'
              else '#dfa05c' if n.startswith('always') else '#8768a6' if 'frozen' in n
              else '#5b86a5' for n in names]
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.2), constrained_layout=True)
    measures = [('accuracy', 'Accuracy (%)', 100), ('input_token_reduction_ratio', 'Input token reduction (%)', 100),
                ('harmful_accepted_count', 'Correct original -> wrong selection', 1)]
    for ax, (metric, title, scale) in zip(axes, measures):
        values = [scale * policies[n][metric] for n in names]
        bars = ax.bar(np.arange(len(names)), values, color=colors, width=.72)
        ax.bar_label(bars, labels=[f'{v:.1f}' if scale == 100 else str(int(v)) for v in values], padding=3, fontsize=8)
        ax.set_xticks(np.arange(len(names)), [plot_label(n) + (' *' if n == primary else '') for n in names], rotation=42, ha='right', fontsize=8)
        ax.set_ylim(0, max(105 if metric == 'accuracy' else 10, max(values) * 1.18 + 1))
        ax.set_ylabel(title)
        ax.grid(axis='y', alpha=.18)
        ax.set_axisbelow(True)
    fig.suptitle(f"Same fresh synthetic test (n={metrics['test_tasks']}); * calibration-selected hybrid", fontsize=13)
    fig.savefig(target / 'policy_comparison.png', dpi=180)
    plt.close(fig)

    m, criteria = policies[primary], metrics['success_criteria']
    checked = metrics['primary_success_criteria_observed']
    specs = [
        ('Input token reduction (%)', 100*m['input_token_reduction_ratio'], 100*criteria['input_token_reduction_ratio_min'], checked['saves_at_least_20_percent'], 'at least', 'input_token_reduction_ratio_95_ci'),
        ('Accuracy change vs original (pp)', 100*m['accuracy_delta'], 100*criteria['accuracy_delta_min'], checked['accuracy_delta_at_least_minus_2_5pp'], 'at least', 'accuracy_delta_95_ci'),
        ('Harmful selections / all tasks (%)', 100*m['harmful_accepted_per_task'], 100*criteria['harmful_accepted_per_task_max'], checked['harmful_acceptance_at_most_5_percent'], 'at most', None),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 3.4), constrained_layout=True)
    ci = m.get('paired_group_bootstrap', {})
    for ax, (title, value, goal, passed, direction, ci_key) in zip(axes, specs):
        color = '#137a68' if passed else '#ce8b36'
        ax.barh([0], [value], color=color, height=.38)
        bounds = [value, goal, 0.]
        if ci_key and ci_key in ci:
            lo, hi = [100*v for v in ci[ci_key]]
            ax.plot([lo, hi], [0, 0], color='#263b49', linewidth=2)
            ax.plot([lo, hi], [0, 0], '|', color='#263b49', markersize=12)
            bounds += [lo, hi]
        width = max(max(bounds)-min(bounds), 5)
        ax.set_xlim(min(bounds)-.15*width, max(bounds)+.25*width)
        ax.axvline(goal, color='#b24b41', linestyle='--', linewidth=1.3)
        ax.set_yticks([])
        ax.set_ylim(-.6, .6)
        ax.set_xlabel(title)
        ax.set_title(f"Observed {value:.2f}; {direction} {goal:.2f}\n{'MET' if passed else 'NOT MET'}", fontsize=10)
        ax.grid(axis='x', alpha=.15)
        ax.set_axisbelow(True)
    fig.suptitle(f"Primary hybrid: {plot_label(primary)}; observed targets, not guarantees", fontsize=12)
    fig.savefig(target / 'primary_targets.png', dpi=180)
    plt.close(fig)

    types = sorted(metrics['by_task_type'][primary])
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5), constrained_layout=True)
    x = np.arange(len(types))
    comparison = [('original', -.24, '#b9c3cb'), ('always_60', 0., '#dfa05c'), (primary, .24, '#137a68')]
    for name, offset, color in comparison:
        axes[0].bar(x+offset, [100*metrics['by_task_type'][name][t]['accuracy'] for t in types], width=.23, label=plot_label(name), color=color)
    for name, offset, color in [('always_60', -.17, '#dfa05c'), (primary, .17, '#137a68')]:
        axes[1].bar(x+offset, [100*metrics['by_task_type'][name][t]['input_token_reduction_ratio'] for t in types], width=.32, label=plot_label(name), color=color)
    axes[0].set_ylim(0, 108)
    axes[0].set_ylabel('Accuracy (%)')
    axes[1].set_ylabel('Input token reduction (%)')
    for ax in axes:
        ax.set_xticks(x, [t.replace('_', ' ') for t in types], rotation=38, ha='right', fontsize=9)
        ax.legend(fontsize=8, loc='lower left' if ax is axes[0] else 'upper right')
        ax.grid(axis='y', alpha=.18)
        ax.set_axisbelow(True)
    fig.suptitle('Held-out source and wording families within known structures', fontsize=13)
    fig.savefig(target / 'structure_breakdown.png', dpi=180)
    plt.close(fig)


def blocks_for(metrics, frozen, verification, audit_name):
    primary = metrics['primary_policy']
    p, m = metrics['policies'], metrics['policies'][primary]
    n, met = metrics['test_tasks'], metrics['all_predeclared_primary_criteria_met']
    blocks = []
    add = lambda kind, value: blocks.append((kind, value))
    add('h1', 'CacheScope V3 实验结果')
    add('p', f"最终主方案是“{label(primary)}”，由新校准集选出，并在这 {n} 个新测试任务、{metrics['test_groups']} 个来源／措辞模板组上评估。它是神经风险分数与显式JSON结构检查组成的混合方案，不能称为纯MLP。")
    add('status', f"{'本批观测达到' if met else '本批观测尚未同时达到'}三项预定目标：准确率 {pct(m['accuracy'])}，完整输入token节省 {pct(m['input_token_reduction_ratio'])}；{m['harmful_accepted_count']}/{n} 题由原本正确变为答错（{pct(m['harmful_accepted_per_task'])}）。达到经验目标不代表未来数据的风险保证。")
    add('h2', '新训练、校准和主方案的归属')
    counts = frozen['counts']
    add('p', f"旧V1/V2的 {counts['development']} 条任务记录现在全部是开发数据，其中旧的50条长度派生记录仍共享来源；原有测试不再为V3提供独立验证。内部按组划分选择训练轮数后，在旧开发数据上重训权重；新的 {counts['calibration']} 题只负责阈值和主方案选择；最后 {counts['test']} 题不参与拟合或选择。")
    if primary == 'hybrid_mlp_ensemble':
        add('p', '本次新训练的54维MLP集成经校准入选主方案。三个预定种子520/521/522的风险分数逐候选取最大值，再加JSON结构检查；没有在测试集上选择表现最好的单一种子。完整对照仍保留各单种子、纯集成、冻结V2和仅结构检查。')
    else:
        add('p', '本次校准选中了冻结V2权重加新门槛和JSON结构检查的方案。新训练的54维MLP集成虽已完成训练，但没有入选主方案；因此主结果的改善不能归因于新训练权重。新训方案和三个单种子的真实结果仍完整保留，不能用较好测试种子替代主方案。')
    add('p', f"校准预先按有害接受上限 {', '.join(pct(x) for x in frozen['predeclared_harm_tiers'])} 从严到宽检查，首先选择存在至少20%输入token节省的档，再在该档比较预定混合候选；本次冻结采用的经验上限为 {pct(frozen['calibration_epsilon_empirical'])}。同时要求校准准确率相对原文下降不超过1.25个百分点。最终测试的三项目标保持20%节省、准确率下降不超过2.5个百分点、有害接受不超过5%。")
    cal = frozen['primary_calibration_metrics']
    add('table', (['主方案在哪一阶段', '任务数', '准确率', '输入token节省', '有害接受次数'], [
        ['新校准集（用于选择）', cal['n_tasks'], pct(cal['accuracy']), pct(cal['input_token_reduction_ratio']), cal['harmful_accepted_count']],
        ['新测试集（用于最终评价）', n, pct(m['accuracy']), pct(m['input_token_reduction_ratio']), m['harmful_accepted_count']],
    ]))
    add('p', '结构检查只拒绝“原文是合法JSON容器、压缩后无法解析为JSON容器”的候选；其余风险由学习到的分数决定。它不是语义验证器。所有特征与结构判断只使用原文、问题、候选文本和token数，不读取参考答案、实际回答或正确性。')
    add('h2', f"同一新测试集上的全部 {len(p)} 项策略")
    add('table', table_policies(metrics, primary))
    add('p', '每项策略按任务计一次准确率，原文答错的题仍在分母内。“原文正确→答错”和“原文错误→答对”分别报告，不能仅用两者相抵后的净准确率隐藏压缩退化。不同策略的压缩程度与校准门槛不同，不能把不同token预算下的准确率差解释为预算完全相同时的因果优势。')
    add('image', ('results/figures/policy_comparison.png', '同一新测试集上的已冻结策略；星号表示校准选定的混合主方案'))
    add('h2', '三项目标与不确定性')
    checks = metrics['primary_success_criteria_observed']
    add('table', (['预定目标', '主方案实测', '本批判断'], [
        ['输入token节省≥20%', pct(m['input_token_reduction_ratio']), '达到' if checks['saves_at_least_20_percent'] else '未达到'],
        ['准确率相对原文下降≤2.5个百分点', pp(m['accuracy_delta']), '达到' if checks['accuracy_delta_at_least_minus_2_5pp'] else '未达到'],
        ['有害接受占全部任务≤5%', pct(m['harmful_accepted_per_task']), '达到' if checks['harmful_acceptance_at_most_5_percent'] else '未达到'],
    ]))
    ci = m['paired_group_bootstrap']
    add('p', f"主方案相对原文准确率差为 {pp(m['accuracy_delta'])}，按模板组配对bootstrap得到的95%区间为 {interval(ci['accuracy_delta_95_ci'], True)}；输入token节省区间为 {interval(ci['input_token_reduction_ratio_95_ci'])}。点估计达标不代表已经证明统计上的非劣性，也不意味着在未来任务中保证同样的风险上限。")
    add('image', ('results/figures/primary_targets.png', '预定目标与实测值；准确率差和token节省的黑线为模板组bootstrap区间'))
    compared = []
    for key, value in metrics.get('paired_comparisons', {}).items():
        other = key.removeprefix('primary_vs_')
        compared.append([label(other), value['a_correct_b_wrong'], value['a_wrong_b_correct'], pp(value['accuracy_difference_a_minus_b']), interval(value['accuracy_difference_group_bootstrap_95_ci'], True), pvalue(value['exact_mcnemar_two_sided_p'])])
    if compared:
        add('table', (['主方案对照', '主方案对/对照错', '主方案错/对照对', '准确率差', '组bootstrap 95%区间', '逐题McNemar p'], compared))
    add('p', 'McNemar值是逐任务精确二项检验，未校正模板内相关性；模板组bootstrap也受有限组数影响。比较没有多重检验校正，不能据单一p值宣称普遍优越。极小p值按“<0.0001”展示，完整精确值保留在指标JSON。')
    add('h2', '纯神经、结构规则与混合策略的区别')
    for base, hybrid in [('mlp_ensemble', 'hybrid_mlp_ensemble'), ('frozen_v2_mlp', 'hybrid_frozen_v2_mlp')]:
        a, b = p[base], p[hybrid]
        add('p', f"{label(base)}：准确率 {pct(a['accuracy'])}、token节省 {pct(a['input_token_reduction_ratio'])}、有害接受 {a['harmful_accepted_count']} 次；{label(hybrid)}：对应为 {pct(b['accuracy'])}、{pct(b['input_token_reduction_ratio'])}、{b['harmful_accepted_count']} 次。这些策略各自采用已冻结的校准门槛；差异反映整个策略，不能全部归因于加了一个检查。")
    only = p['guard_only']
    add('p', f"仅JSON结构检查的准确率为 {pct(only['accuracy'])}、token节省为 {pct(only['input_token_reduction_ratio'])}、有害接受 {only['harmful_accepted_count']} 次。这个对照用于区分显式规则与神经风险门控，不能只展示混合方案而声称全部收益都来自深度学习。")
    seed = metrics.get('seed_variation', {})
    if seed:
        add('p', f"三个预定纯MLP种子的测试准确率均值为 {pct(seed['accuracy_mean'])}，样本标准差为 {100*seed['accuracy_std']:.2f} 个百分点。三个种子均在完整表中，主方案身份由校准记录决定；最大值集成也不是测试后挑选最好种子。")
    cm = metrics.get('conditional_classifier_metrics', {})
    if cm:
        rows = [[label(name), v['n'], v['positives'], number(v.get('average_precision')), number(v.get('roc_auc'))] for name, v in cm.items()]
        add('table', (['条件风险分数', '原文正确候选数', '退化候选数', 'AP', 'ROC-AUC'], rows))
        add('p', '风险分类的退化标签定义在原文答对的候选上；分数是未校准的条件风险排序，不是任意请求的绝对错误概率。混合分数含结构拒绝编码，相关AP/AUC也不能当成纯神经模型指标。')
    add('h2', '长度与结构分项')
    length_rows = []
    for band in sorted(metrics['by_length'][primary]):
        for name in dict.fromkeys(['original', 'always_60', 'frozen_v2_mlp', 'mlp_ensemble', primary]):
            value = metrics['by_length'][name][band]
            length_rows.append(['较长（≥400上下文token）' if band == 'long' else '较短（<400上下文token）', label(name), value['n_tasks'], pct(value['accuracy']), pct(value['input_token_reduction_ratio']), value['harmful_accepted_count']])
    add('table', (['长度层', '策略', '任务数', '准确率', '输入token节省', '有害接受'], length_rows))
    structure_rows = []
    for kind in sorted(metrics['by_task_type'][primary]):
        value, original = metrics['by_task_type'][primary][kind], metrics['by_task_type']['original'][kind]
        structure_rows.append([kind, value['n_tasks'], pct(original['accuracy']), pct(value['accuracy']), pct(value['input_token_reduction_ratio']), value['harmful_accepted_count'], value['improvement_count']])
    add('table', (['结构', '任务数', '原文准确率', '主方案准确率', '输入token节省', '有害接受', '改善'], structure_rows))
    add('image', ('results/figures/structure_breakdown.png', '已见高层结构中的新来源与留出措辞家族；不据分项结果重新调门槛'))
    add('p', '这些都是合成英文姓名、日期、预算事实。八类高层结构在开发数据中已见，测试留出新的来源与措辞／布局家族，不能称为完全新结构的分布外验证。400-token分层只表示本数据内的相对长度，不等于真实长文档基准。单一小型回答模型上的观测也不能推广为其他模型、语言或真实文档的保证。')
    add('h2', '核验、案例与复现')
    if verification is None:
        add('p', '完整核验JSON尚未提供；当前报告不能据此声称原始回答、标签、token和哈希的最终核验已经通过。')
    else:
        passed = verification.get('pass', verification.get('passed', None)) if isinstance(verification, dict) else None
        add('p', f"核验文件的整体状态：{'通过' if passed is True else '未通过或未明确标注，须检查原文件'}。")
        if isinstance(verification, dict):
            scalars = [[key, str(value)] for key, value in verification.items() if isinstance(value, (str, int, float, bool)) or value is None]
            add('table', (['核验记录项', '实测值'], scalars))
        if audit_name:
            add('link', (audit_name, '完整核验JSON'))
    for path, title in [('results/metrics.json', '全部真实指标JSON'), ('results/frozen_selection.json', '测试前冻结的主方案、门槛与哈希'), ('results/policy_results.csv', '完整策略CSV'), ('results/task_policy_decisions.jsonl', '逐任务已冻结策略决策'), ('docs/DATA_CARD.md', '数据卡与范围'), ('docs/FEATURES_V3.md', '54维特征与结构守门说明'), ('docs/FAILURE_ANALYSIS.md', '真实失败与改善案例')]:
        add('link', (path, title))
    add('p', '数据生成、代码与文档使用了AI辅助。标签来自实际本地模型回答与预定事实的匹配。token节省使用完整输入提示词计数，不能直接解释为API费用、订阅额度或端到端延迟节省。训练、校准和测试的原始观测均应保留，未达目标的结果也不应隐藏。')
    return blocks


def markdown(blocks):
    out = []
    for kind, value in blocks:
        if kind in ('h1', 'h2'):
            out.append(('# ' if kind == 'h1' else '## ') + value)
        elif kind == 'table':
            heads, rows = value
            cell = lambda x: str(x).replace('|', '\\|').replace('\n', ' ')
            out.append('\n'.join(['| ' + ' | '.join(map(cell, heads)) + ' |', '| ' + ' | '.join(['---']*len(heads)) + ' |'] + ['| ' + ' | '.join(map(cell, row)) + ' |' for row in rows]))
        elif kind == 'image':
            out.append(f'![{value[1]}]({value[0]})')
        elif kind == 'link':
            out.append(f'[{value[1]}]({value[0]})')
        elif kind == 'status':
            out.append('**' + value + '**')
        else:
            out.append(value)
    return '\n\n'.join(out) + '\n'


def html_document(blocks, success):
    esc = lambda x: html.escape(str(x), quote=True)
    parts = []
    for kind, value in blocks:
        if kind in ('h1', 'h2'):
            parts.append(f'<{kind}>{esc(value)}</{kind}>')
        elif kind == 'status':
            parts.append(f'<div class="status {"met" if success else "unmet"}">{esc(value)}</div>')
        elif kind == 'table':
            heads, rows = value
            parts.append('<div class="table-scroll"><table><thead><tr>' + ''.join(f'<th>{esc(x)}</th>' for x in heads) + '</tr></thead><tbody>' + ''.join('<tr>' + ''.join(f'<td>{esc(x)}</td>' for x in row) + '</tr>' for row in rows) + '</tbody></table></div>')
        elif kind == 'image':
            parts.append(f'<figure><img src="{esc(value[0])}" alt="{esc(value[1])}"><figcaption>{esc(value[1])}</figcaption></figure>')
        elif kind == 'link':
            parts.append(f'<p class="source"><a href="{esc(value[0])}">{esc(value[1])}</a></p>')
        else:
            parts.append(f'<p>{esc(value)}</p>')
    style = '''body{margin:0;background:#f2f6f7;color:#20333e;font:16px/1.75 "Microsoft YaHei","PingFang SC",Arial,sans-serif}main{max-width:1180px;margin:32px auto;padding:36px 42px;background:#fff;border-radius:14px;box-shadow:0 5px 24px #20333e0b}h1{font-size:29px;line-height:1.35;color:#104a45}h2{font-size:22px;margin-top:36px;padding-top:18px;border-top:1px solid #e0e8e9}.status{padding:17px 21px;border-radius:7px;font-weight:600}.met{background:#e5f2ed;border-left:5px solid #137a68}.unmet{background:#fff1dc;border-left:5px solid #ce8b36}table{width:100%;min-width:740px;border-collapse:collapse;font-size:14px}th,td{text-align:left;padding:10px 12px;border-bottom:1px solid #e2e9eb}th{background:#edf4f3;white-space:nowrap}tbody tr:nth-child(even){background:#f8fafb}.table-scroll{overflow:auto;border:1px solid #e2e9eb;border-radius:7px;margin:20px 0}img{width:100%;height:auto}figure{margin:25px 0}figcaption{font-size:13px;color:#657b86;text-align:center}.source{display:inline-block;margin-right:18px}a{color:#137a68}@media(max-width:700px){main{margin:0;padding:22px 16px;border-radius:0}h1{font-size:25px}h2{font-size:20px}}'''
    return '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>CacheScope V3 实验结果</title><style>' + style + '</style><main>' + '\n'.join(parts) + '</main></html>\n'


def build(root=ROOT, verification_override=None):
    root = Path(root).resolve()
    metrics = read_json(root/'results/metrics.json')
    frozen = read_json(root/'results/frozen_selection.json')
    primary = validate_inputs(metrics, frozen)
    audit = Path(verification_override).resolve() if verification_override else root/'results/verification.json'
    verification = read_json(audit) if audit.exists() else None
    try:
        audit_name = audit.relative_to(root).as_posix()
    except ValueError:
        audit_name = None
    make_figures(metrics, frozen, root/'results/figures')
    blocks = blocks_for(metrics, frozen, verification, audit_name)
    (root/'RESULTS.md').write_text(markdown(blocks), encoding='utf-8')
    (root/'实验结果.html').write_text(html_document(blocks, metrics['all_predeclared_primary_criteria_met']), encoding='utf-8')
    print(json.dumps({'built':['RESULTS.md','实验结果.html','results/figures/policy_comparison.png','results/figures/primary_targets.png','results/figures/structure_breakdown.png'], 'primary_policy':primary, 'primary_is_hybrid':True, 'policies_included':len(metrics['policies']), 'input':'observed frozen-policy metrics only'}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--verification', type=Path)
    args = parser.parse_args()
    build(args.root, args.verification)
