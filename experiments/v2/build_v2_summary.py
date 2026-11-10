"""Render V2's locked, observed results; never train, select, or re-label.

Inputs are metrics.json, frozen_model_selection.json, and an optional audit
JSON. The report always describes predeclared seed 520, including failures.
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent

LABELS = {
    "original": "原文", "always_60": "固定保留60%", "always_30": "固定保留30%",
    "simple_rule_gate": "简单规则门控", "frozen_v1_mlp": "冻结V1 MLP",
    "mlp": "V2 MLP（主seed 520）", "lr": "V2 Logistic Regression", "rf": "V2 Random Forest",
    "mlp_seed_521": "V2 MLP（seed 521）", "mlp_seed_522": "V2 MLP（seed 522）",
    "mlp_v1features": "旧20维特征＋V2数据MLP",
}
PLOT_LABELS = {
    "original": "Original", "always_60": "Fixed keep 60%", "always_30": "Fixed keep 30%",
    "simple_rule_gate": "Rule gate", "frozen_v1_mlp": "Frozen V1 MLP",
    "mlp": "V2 MLP (520)", "lr": "V2 LR", "rf": "V2 RF",
    "mlp_v1features": "V1 features / V2 data",
}
ORDER = ["original", "always_60", "always_30", "simple_rule_gate", "frozen_v1_mlp",
         "lr", "rf", "mlp", "mlp_v1features", "mlp_seed_521", "mlp_seed_522"]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def percent(value, digits=2):
    return "—" if value is None else f"{100 * value:.{digits}f}%"


def pp(value, digits=2):
    return "—" if value is None else f"{100 * value:+.{digits}f}个百分点"


def number(value, digits=3):
    return "—" if value is None else f"{value:.{digits}f}"


def interval(values, percentage_points=False):
    if values is None:
        return "未提供"
    if percentage_points:
        return f"[{100 * values[0]:+.2f}, {100 * values[1]:+.2f}] 个百分点"
    return f"[{100 * values[0]:.2f}%, {100 * values[1]:.2f}%]"


def policy_table(policies, order=ORDER):
    headers = ["策略", "准确率", "输入token节省", "采用压缩", "原本正确→答错", "原本答错→正确"]
    rows = [[LABELS[name], percent(policies[name]["accuracy"]),
             percent(policies[name]["input_token_reduction_ratio"]),
             str(policies[name]["accepted_count"]), str(policies[name]["harmful_accepted_count"]),
             str(policies[name]["improvement_count"])] for name in order if name in policies]
    return headers, rows


def make_figures(metrics, target):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    target.mkdir(parents=True, exist_ok=True)
    policies = metrics["policies"]
    names = [n for n in ("original", "always_60", "always_30", "frozen_v1_mlp", "lr", "rf", "mlp") if n in policies]
    labels = [PLOT_LABELS[n] for n in names]
    colors = ["#b7c2cd" if n == "original" else "#e4a35c" if n.startswith("always")
              else "#b390c7" if n == "frozen_v1_mlp" else "#197568" if n == "mlp" else "#5986a8" for n in names]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.9), constrained_layout=True)
    for ax, metric, title in zip(axes, ["accuracy", "input_token_reduction_ratio"],
                                ["Task accuracy (%)", "Full input prompt token reduction (%)"]):
        values = [100 * policies[n][metric] for n in names]
        bars = ax.bar(np.arange(len(names)), values, color=colors, width=.7)
        ax.bar_label(bars, labels=[f"{v:.1f}" for v in values], padding=3, fontsize=9)
        ax.set_xticks(np.arange(len(names)), labels, rotation=30, ha="right")
        ax.set_ylim(0, max(100 if metric == "accuracy" else 15, max(values) + 12))
        ax.set_ylabel(title)
        ax.grid(axis="y", alpha=.18)
        ax.set_axisbelow(True)
    fig.suptitle(f"Locked policies on the same fresh synthetic test (n={metrics['test_tasks']})", fontsize=13)
    fig.savefig(target / "policy_comparison.png", dpi=180)
    plt.close(fig)

    names = ["always_60", "frozen_v1_mlp", "mlp"]
    values = [[100 * policies[n][metric] for n in names]
              for metric in ("accuracy", "input_token_reduction_ratio")]
    fig, axes = plt.subplots(1, 3, figsize=(12.7, 4.4), constrained_layout=True)
    colors = ["#e4a35c", "#b390c7", "#197568"]
    for i, ax in enumerate(axes):
        raw = values[i] if i < 2 else [policies[n]["harmful_accepted_count"] for n in names]
        bars = ax.bar(np.arange(3), raw, color=colors, width=.65)
        ax.bar_label(bars, labels=[f"{v:.1f}" if i < 2 else str(v) for v in raw], padding=4)
        ax.set_xticks(np.arange(3), [PLOT_LABELS[n] for n in names], rotation=20, ha="right")
        ax.set_ylabel(["Task accuracy (%)", "Input token reduction (%)", "Correct original -> wrong selection (count)"][i])
        ax.set_ylim(0, max(100 if i == 0 else 10, max(raw) * 1.22 + 1))
        ax.grid(axis="y", alpha=.18)
        ax.set_axisbelow(True)
    fig.suptitle(f"V1 and V2 evaluated on identical new tasks (n={metrics['test_tasks']}); V2 seed 520", fontsize=13)
    fig.savefig(target / "same_fresh_test_v1_v2.png", dpi=180)
    plt.close(fig)

    types = sorted(metrics["by_task_type"]["mlp"])
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), constrained_layout=True)
    x = np.arange(len(types))
    for offset, name, color in [(-.24, "original", "#b7c2cd"), (0, "always_60", "#e4a35c"), (.24, "mlp", "#197568")]:
        axes[0].bar(x + offset, [100 * metrics["by_task_type"][name][t]["accuracy"] for t in types],
                    width=.23, label=PLOT_LABELS[name], color=color)
    axes[0].set_ylabel("Task accuracy (%)")
    axes[0].set_ylim(0, 108)
    axes[0].legend(fontsize=8, loc="lower left")
    for offset, name, color in [(-.17, "always_60", "#e4a35c"), (.17, "mlp", "#197568")]:
        axes[1].bar(x + offset, [100 * metrics["by_task_type"][name][t]["input_token_reduction_ratio"] for t in types],
                    width=.32, label=PLOT_LABELS[name], color=color)
    axes[1].set_ylabel("Full input token reduction (%)")
    axes[1].legend(fontsize=8, loc="upper right")
    for ax in axes:
        ax.set_xticks(x, [t.replace("_", " ") for t in types], rotation=35, ha="right")
        ax.grid(axis="y", alpha=.18)
        ax.set_axisbelow(True)
    fig.suptitle("Fresh-test breakdown: observed structure families, no threshold changes", fontsize=13)
    fig.savefig(target / "structure_breakdown.png", dpi=180)
    plt.close(fig)


def load_verification(root, override=None):
    candidates = [Path(override)] if override else [root / "results/verification.json", root / "results/verification_report.json"]
    for path in candidates:
        if path.exists():
            return path, read_json(path)
    return None, None


def make_blocks(metrics, frozen, verification_path, verification):
    p = metrics["policies"]
    m = p["mlp"]
    n = metrics["test_tasks"]
    met = metrics["all_predeclared_primary_criteria_met"]
    blocks = []
    add = lambda kind, value: blocks.append((kind, value))
    add("h1", "CacheScope V2 实验结果")
    add("p", f"主模型固定为 seed 520 的 MLP。本次在 {n} 个新测试任务、{metrics['test_groups']} 个模板组上评估；所有模型和门槛在读取测试观测前已冻结。" )
    add("status", f"{'达到' if met else '尚未同时达到'}预先设定的三项主模型条件。实测准确率 {percent(m['accuracy'])}，输入 token 节省 {percent(m['input_token_reduction_ratio'])}，{m['harmful_accepted_count']} 个任务由原本正确变为答错（占全部任务 {percent(m['harmful_accepted_per_task'])}）。")
    add("h2", "这次改了什么")
    add("p", f"研究问题仍是：在回答模型运行前，用轻量风险模型决定是否接受固定压缩器的输出。压缩器仍产生保留60%与30%上下文的两档候选；门控选择通过门槛的最短候选，否则保留原文。V2 使用 {len(frozen['feature_names'])} 维回答前特征，增加与问题相关的证据保留信息，并扩充训练数据中的结构和长度。没有用参考答案、实际回答或正确性标签作为推断特征。")
    counts = frozen["counts"]
    add("p", f"训练集有 {counts['train']['tasks']} 条任务记录，新的验证集有 {counts['validation']['tasks']} 条任务，新测试集有 {n} 条任务。全部旧V1测试、结构留出和长度压力数据现已归入开发数据；旧数据上的表现不能作为V2的独立验证。旧长度派生记录也不能按新增独立来源计算。")
    add("p", "验证集负责早停和门槛选择：在经验有害接受比例≤2.5%、准确率相对原文下降≤2.5个百分点的约束下，最大化输入token节省。该经验约束不保证新测试集达到同样水平。主种子520预先固定；521、522只用于报告稳定性。")
    add("h2", "同一批新测试任务上的完整对照")
    add("table", policy_table(p))
    add("p", "token节省按完整输入提示词的实际token总量计算；原本答错的任务仍计入策略准确率。表中“原本正确→答错”是压缩造成的观测退化，“原本答错→正确”是采用压缩后观测到的改善。两者相抵后的准确率变化不能代替有害接受数量。")
    add("image", ("results/figures/policy_comparison.png", "同一新测试集的策略准确率和输入token节省"))
    add("h2", "V1 与 V2：使用同一新测试集比较")
    v1 = p["frozen_v1_mlp"]
    fixed = p["always_60"]
    add("p", f"冻结V1模型在这 {n} 个新任务上的准确率为 {percent(v1['accuracy'])}、输入token节省为 {percent(v1['input_token_reduction_ratio'])}、有害接受 {v1['harmful_accepted_count']} 次。V2主MLP对应为 {percent(m['accuracy'])}、{percent(m['input_token_reduction_ratio'])}、{m['harmful_accepted_count']} 次。V2相对该V1的准确率变化为 {pp(m['accuracy'] - v1['accuracy'])}，token节省变化为 {pp(m['input_token_reduction_ratio'] - v1['input_token_reduction_ratio'])}。")
    add("p", f"固定保留60%策略的准确率为 {percent(fixed['accuracy'])}、输入token节省为 {percent(fixed['input_token_reduction_ratio'])}、有害接受 {fixed['harmful_accepted_count']} 次。V2主MLP相对它的准确率变化为 {pp(m['accuracy'] - fixed['accuracy'])}，token节省变化为 {pp(m['input_token_reduction_ratio'] - fixed['input_token_reduction_ratio'])}。节省比例不相等时，应同时比较准确率与token，不能直接解释为完全相同预算下的因果优势。")
    add("p", "这里没有拿V1旧100题的准确率与V2新测试准确率直接比较；三个策略均运行于同一批已冻结的新任务。V1在这里保留旧权重、旧特征定义和旧门槛，不重新调参。")
    add("image", ("results/figures/same_fresh_test_v1_v2.png", "固定压缩、冻结V1与V2在相同新任务上的比较"))
    add("h2", "预定目标与不确定性")
    checks = metrics["primary_success_criteria_observed"]
    targets = [
        ["输入token节省≥20%", percent(m["input_token_reduction_ratio"]), "达到" if checks["saves_at_least_20_percent"] else "未达到"],
        ["准确率相对原文下降≤2.5个百分点", pp(m["accuracy_delta"]), "达到" if checks["accuracy_delta_at_least_minus_2_5pp"] else "未达到"],
        ["有害接受占全部任务≤5%", percent(m["harmful_accepted_per_task"]), "达到" if checks["harmful_acceptance_at_most_5_percent"] else "未达到"],
    ]
    add("table", (["预先设定的条件", "主MLP实测", "判断"], targets))
    ci = m["paired_group_bootstrap"]
    add("p", f"主MLP相对原文的准确率变化为 {pp(m['accuracy_delta'])}，按模板组重采样的95%区间为 {interval(ci['accuracy_delta_95_ci'], True)}；输入token节省的95%区间为 {interval(ci['input_token_reduction_ratio_95_ci'])}。区间反映这批合成模板任务的不确定性，不代表真实用户总体或其他回答模型。")
    paired_rows = []
    for other in ("original", "always_60", "frozen_v1_mlp", "mlp_v1features", "rf"):
        c = metrics["paired_comparisons"][f"mlp_vs_{other}"]
        paired_rows.append([LABELS[other], str(c["a_correct_b_wrong"]), str(c["a_wrong_b_correct"]),
                            pp(c["accuracy_difference_a_minus_b"]),
                            interval(c["accuracy_difference_group_bootstrap_95_ci"], True),
                            number(c["exact_mcnemar_two_sided_p"], 4)])
    add("table", (["主MLP对照", "MLP对/对照错", "MLP错/对照对", "准确率差", "组bootstrap 95%区间", "逐题McNemar p"], paired_rows))
    add("p", "McNemar列使用逐任务精确二项检验，没有校正模板内相关性；优先结合模板组bootstrap区间理解。以上比较均如实报告，不用测试结果更换模型、门槛或主种子；也没有针对多重比较做校正。")
    add("h2", "MLP、传统模型、特征消融与种子稳定性")
    rf = p["rf"]
    ab = p["mlp_v1features"]
    add("p", f"Random Forest准确率为 {percent(rf['accuracy'])}、token节省为 {percent(rf['input_token_reduction_ratio'])}、有害接受 {rf['harmful_accepted_count']} 次；主MLP对应为 {percent(m['accuracy'])}、{percent(m['input_token_reduction_ratio'])}、{m['harmful_accepted_count']} 次。该结果应按实际取舍解读，不能因为课程包含深度学习就声称神经网络一定优于传统模型。")
    add("p", f"保持V2扩展开发数据、改回V1旧20维特征后，MLP准确率为 {percent(ab['accuracy'])}、token节省为 {percent(ab['input_token_reduction_ratio'])}、有害接受 {ab['harmful_accepted_count']} 次。这个消融使用同样的数据、架构、主种子和验证规则，但各自选择验证门槛，因此它比较整个特征加门槛方案，不能证明某单一特征的独立因果作用。")
    seed = metrics["seed_variation"]
    add("p", f"三个预定种子520/521/522的准确率均值为 {percent(seed['accuracy_mean'])}，样本标准差为 {100 * seed['accuracy_std']:.2f} 个百分点；token节省均值为 {percent(seed['input_token_reduction_mean'])}，样本标准差为 {100 * seed['input_token_reduction_std']:.2f} 个百分点。三个种子的单独结果均在上表，主结果始终使用520。")
    cm = metrics["conditional_classifier_metrics"]
    cr = []
    for name in ("mlp", "lr", "rf", "mlp_v1features", "frozen_v1_mlp"):
        v = cm[name]
        cr.append([LABELS[name], str(v["n"]), str(v["positives"]), number(v["average_precision"]), number(v["roc_auc"])])
    add("table", (["条件风险分类器", "原文正确候选数", "退化候选数", "AP", "ROC-AUC"], cr))
    add("p", "风险分类只在原文回答正确的候选上定义“压缩导致退化”；模型输出是该条件下学到的风险排序分数，不是对任意请求都已校准的绝对失败概率。策略表仍保留全部任务。")
    add("h2", "长度与结构表现")
    length_rows = []
    for band in sorted(metrics["by_length"]["mlp"]):
        for name in ("original", "always_60", "frozen_v1_mlp", "mlp"):
            v = metrics["by_length"][name][band]
            length_rows.append(["较长（≥400上下文token）" if band == "long" else "较短（<400上下文token）",
                                LABELS[name], str(v["n_tasks"]), percent(v["accuracy"]),
                                percent(v["input_token_reduction_ratio"]), str(v["harmful_accepted_count"])])
    add("table", (["文本长度", "策略", "任务数", "准确率", "token节省", "有害接受"], length_rows))
    add("image", ("results/figures/structure_breakdown.png", "新测试集按结构分类的准确率与token节省"))
    add("p", "八种高层结构在开发数据中都出现过，新的测试任务与模板组独立。因此本次检验的是已见结构下的新来源、新措辞及不同长度任务，不能声称已经解决完全新结构的分布外泛化。较长文本的400-token分层也不等于真实长文档基准。")
    add("h2", "审计状态与复现材料")
    if verification is None:
        add("p", "当前尚未发现完整独立审计JSON；模型选择冻结记录与评估结果已生成，独立数据／响应核验状态应以最终审计文件为准。")
    else:
        if isinstance(verification, dict):
            scalars = [[str(k), str(v)] for k, v in verification.items() if isinstance(v, (str, int, float, bool)) or v is None]
            if scalars:
                add("table", (["审计文件记录项", "实际值"], scalars[:20]))
            else:
                add("p", "已读取独立审计JSON；详细检查项见链接文件。")
        else:
            add("p", "已读取独立审计JSON；详细检查项见链接文件。")
        try:
            audit_relative = str(verification_path.relative_to(ROOT)).replace("\\", "/")
            add("link", (audit_relative, "独立审计JSON"))
        except ValueError:
            add("p", f"审计来源：{verification_path.name}")
    add("link", ("results/metrics.json", "完整真实指标JSON"))
    add("link", ("results/frozen_model_selection.json", "测试前冻结的模型／门槛与哈希"))
    add("link", ("results/policy_results.csv", "策略结果CSV"))
    add("link", ("results/task_policy_decisions.jsonl", "逐任务策略决策"))
    add("p", "本项目数据与代码生成使用了AI辅助。数据为人为设定事实的合成任务；标签来自实际本地回答模型输出与预定答案的匹配。结果只支持这套任务、压缩器和回答模型的观测结论。它不直接证明真实文档、其他语言、其他大模型、API费用或端到端延迟上的收益。")
    return blocks


def markdown(blocks):
    out = []
    for kind, value in blocks:
        if kind in ("h1", "h2"):
            out.append(("# " if kind == "h1" else "## ") + value)
        elif kind == "table":
            headers, rows = value
            cell = lambda v: str(v).replace("|", "\\|").replace("\n", " ")
            out.append("\n".join(["| " + " | ".join(map(cell, headers)) + " |",
                                  "| " + " | ".join(["---"] * len(headers)) + " |"] +
                                 ["| " + " | ".join(map(cell, r)) + " |" for r in rows]))
        elif kind == "image":
            out.append(f"![{value[1]}]({value[0]})")
        elif kind == "link":
            out.append(f"[{value[1]}]({value[0]})")
        elif kind == "status":
            out.append("**" + value + "**")
        else:
            out.append(value)
    return "\n\n".join(out) + "\n"


def html_document(blocks, success):
    parts = []
    esc = lambda s: html.escape(str(s), quote=True)
    for kind, value in blocks:
        if kind in ("h1", "h2"):
            parts.append(f"<{kind}>{esc(value)}</{kind}>")
        elif kind == "status":
            parts.append(f'<div class="status {"met" if success else "unmet"}">{esc(value)}</div>')
        elif kind == "table":
            headers, rows = value
            parts.append('<div class="table-scroll"><table><thead><tr>' + "".join(f"<th>{esc(h)}</th>" for h in headers) +
                         "</tr></thead><tbody>" + "".join("<tr>" + "".join(f"<td>{esc(c)}</td>" for c in row) + "</tr>" for row in rows) +
                         "</tbody></table></div>")
        elif kind == "image":
            parts.append(f'<figure><img src="{esc(value[0])}" alt="{esc(value[1])}"><figcaption>{esc(value[1])}</figcaption></figure>')
        elif kind == "link":
            parts.append(f'<p class="source"><a href="{esc(value[0])}">{esc(value[1])}</a></p>')
        else:
            parts.append(f"<p>{esc(value)}</p>")
    return '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">' + \
        '<title>CacheScope V2 实验结果</title><style>' + \
        'body{margin:0;background:#f3f6f8;color:#20313e;font:16px/1.75 "Microsoft YaHei","PingFang SC",Arial,sans-serif}' + \
        'main{max-width:1120px;margin:36px auto;padding:36px 42px;background:white;border-radius:14px;box-shadow:0 5px 25px #20313e0d}' + \
        'h1{font-size:30px;line-height:1.3;margin:0 0 22px;color:#103d40}h2{font-size:22px;margin-top:38px;border-top:1px solid #e2e9ed;padding-top:20px}' + \
        'p{margin:14px 0}.status{padding:18px 22px;margin:22px 0;border-radius:8px;font-weight:600}.met{background:#e5f3ed;border-left:5px solid #197568}.unmet{background:#fff3df;border-left:5px solid #c78c25}' + \
        'table{border-collapse:collapse;min-width:700px;width:100%;font-size:14px}th,td{text-align:left;padding:10px 12px;border-bottom:1px solid #e1e8ed}th{background:#edf4f5;white-space:nowrap}tbody tr:nth-child(even){background:#f8fafb}' + \
        '.table-scroll{overflow:auto;margin:20px 0;border:1px solid #e1e8ed;border-radius:7px}figure{margin:26px 0}img{width:100%;height:auto}figcaption{color:#637581;text-align:center;font-size:13px}.source{display:inline-block;margin-right:20px}a{color:#137367}' + \
        '@media(max-width:700px){main{margin:0;padding:24px 18px;border-radius:0}h1{font-size:25px}h2{font-size:20px}}' + \
        '</style><main>' + "\n".join(parts) + '</main></html>\n'


def build(root=ROOT, verification_override=None):
    global ROOT
    ROOT = Path(root).resolve()
    metrics = read_json(ROOT / "results/metrics.json")
    frozen = read_json(ROOT / "results/frozen_model_selection.json")
    if metrics["primary_seed"] != 520 or frozen["primary_mlp_seed"] != 520:
        raise ValueError("The predeclared primary seed must remain 520")
    if metrics["test_tasks"] != frozen["test_expected_tasks"]:
        raise ValueError("Test-task count differs from frozen protocol")
    verification_path, verification = load_verification(ROOT, verification_override)
    make_figures(metrics, ROOT / "results/figures")
    blocks = make_blocks(metrics, frozen, verification_path, verification)
    (ROOT / "RESULTS.md").write_text(markdown(blocks), encoding="utf-8")
    (ROOT / "实验结果.html").write_text(html_document(blocks, metrics["all_predeclared_primary_criteria_met"]), encoding="utf-8")
    print(json.dumps({"built": ["RESULTS.md", "实验结果.html", "results/figures/policy_comparison.png",
          "results/figures/same_fresh_test_v1_v2.png", "results/figures/structure_breakdown.png"],
          "source": "observed frozen-policy metrics only", "primary_seed": 520}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--verification", type=Path)
    args = parser.parse_args()
    build(args.root, args.verification)
