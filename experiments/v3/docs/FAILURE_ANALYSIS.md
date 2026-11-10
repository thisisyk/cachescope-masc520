# CacheScope V3 真实结果与案例

本报告主方案为 **hybrid_frozen_v2_mlp**。底层使用冻结 V2 MLP 权重；本轮重新校准阈值并加入 JSON 检查。新训练权重仍有独立对照结果，不能声称主方案采用了它们。 该方案结合学习到的风险分数与显式 JSON 结构检查，是混合方案，不能将全部收益归因于重新训练的神经网络。

在 200 个独立新测试任务上，主方案准确率为 75.00%，完整输入 token 节省为 26.23%；原文正确、主方案采用压缩后答错 7 次（占全部任务 3.50%），原文答错后改善 19 次。

预先设定的三项条件：全部达到。逐项判断与所有策略对照以 results/metrics.json 为准。

所有来源均为合成英文 owner/date/budget 事实。八种高层结构已在开发阶段出现，本轮独立留出新来源和措辞／布局家族；不构成真实文档或完全未见结构的泛化证据。

## 案例选择规则

每类按 task_id 升序选最先两例：有害接受、原文答错后改善、原文答错且仍然答错。另外选最先两个 JSON 结构检查在固定主权重、固定主阈值下改变选择的案例。某类没有符合条件的任务时明确记为零，不补造案例。类别间可能出现同一任务。

“去掉结构检查、保持主阈值”的选择仅用于解释这一项检查如何影响当前决策，是对已存候选回答的描述性重放，未作为另一个经独立选择验证的策略。实际纯模型对照采用其自身已锁定阈值，两者同时列出，避免把阈值差异归因于结构检查。

片段优先展示目标实体及其关联记录，可能截断；完整原文和所选上下文保存在 results/case_studies.json。以下只描述可观察的答案、结构和选择，不给出未经实验建立的错误因果机制。

## 原文答对、主方案采用压缩后答错

符合条件的任务数：7；展示 2 例。

### v3-fresh-00050（header）

问题：What is the current approved budget for project Lumen Vela-0050?

Gold：`37443`；原文回答：`37443`；主方案回答：`68699`。

主方案选择 `keep_30`；固定阈值 0.304512918；所选混合分数 0.2770964205265045；底层神经分数 0.2770964205265045。回退原文时所选候选分数为空。

上下文 token：523 → 122；完整输入 token：600 → 199。

| 候选 | 底层神经分数 | 混合分数 | JSON检查通过 | 实际回答正确 |
|---|---:|---:|---|---|
| keep_30 | 0.2770964205265045 | 0.2770964205265045 | True | False |
| keep_60 | 0.27024927735328674 | 0.27024927735328674 | True | False |

原文关键片段：

```text
A project card is headed Lumen Vela-0050. On that card, USD 37443 occupies the approved budget field.
```

主方案所选上下文片段：

```text
The project card for Lumen Peripheral-0050-01 gives owner Ada Ross, release date 2028-03-18, and approved budget USD 54621.
The project card for Lumen Peripheral-0050-09 gives owner Clara Grant, release date 2028-08-22, and approved budget USD 68699.
A project card is headed Lumen Vela-0050.
On that card, USD 37443 occupies the approved budget field.
```

观察范围：这些记录确定了本次输入、候选选择和回答的变化；单个案例不能证明模型为何改变回答，也不能证明该规则对所有未来请求都有效。

### v3-fresh-00122（update）

问题：What is the current approved budget for project Glade Vela-0122?

Gold：`73002`；原文回答：`73002`；主方案回答：`74111`。

主方案选择 `keep_30`；固定阈值 0.304512918；所选混合分数 0.2130006104707718；底层神经分数 0.2130006104707718。回退原文时所选候选分数为空。

上下文 token：172 → 39；完整输入 token：249 → 116。

| 候选 | 底层神经分数 | 混合分数 | JSON检查通过 | 实际回答正确 |
|---|---:|---:|---|---|
| keep_30 | 0.2130006104707718 | 0.2130006104707718 | True | False |
| keep_60 | 0.46128013730049133 | 0.46128013730049133 | True | True |

原文关键片段：

```text
Project Glade Vela-0122 initially carried approved budget USD 74111. The final signed update changes the recorded value to USD 73002.
```

主方案所选上下文片段：

```text
Project Glade Vela-0122 initially carried approved budget USD 74111.
The final signed update changes the recorded value to USD 73002.
```

观察范围：这些记录确定了本次输入、候选选择和回答的变化；单个案例不能证明模型为何改变回答，也不能证明该规则对所有未来请求都有效。

## 原文答错、主方案答对

符合条件的任务数：19；展示 2 例。

### v3-fresh-00021（direct）

问题：What is the current owner for project Drift Vela-0021?

Gold：`Hugo Wells`；原文回答：`Drift Vela-0021`；主方案回答：`Hugo Wells`。

主方案选择 `keep_30`；固定阈值 0.304512918；所选混合分数 0.2510097324848175；底层神经分数 0.2510097324848175。回退原文时所选候选分数为空。

上下文 token：514 → 113；完整输入 token：590 → 189。

| 候选 | 底层神经分数 | 混合分数 | JSON检查通过 | 实际回答正确 |
|---|---:|---:|---|---|
| keep_30 | 0.2510097324848175 | 0.2510097324848175 | True | True |
| keep_60 | 0.31086844205856323 | 0.31086844205856323 | True | False |

原文关键片段：

```text
The signed data card pairs project Drift Vela-0021 with Hugo Wells in its owner field.
```

主方案所选上下文片段：

```text
The project card for Drift Peripheral-0021-06 gives owner Mia Turner, release date 2028-07-17, and approved budget USD 76012.
The signed data card pairs project Drift Vela-0021 with Hugo Wells in its owner field.
The project card for Drift Peripheral-0021-10 gives owner Maya Ross, release date 2028-12-16, and approved budget USD 57730.
```

观察范围：这些记录确定了本次输入、候选选择和回答的变化；单个案例不能证明模型为何改变回答，也不能证明该规则对所有未来请求都有效。

### v3-fresh-00024（direct）

问题：What is the current owner for project Beryl Vela-0024?

Gold：`Finn Turner`；原文回答：`Owen Wells`；主方案回答：`Finn Turner`。

主方案选择 `keep_30`；固定阈值 0.304512918；所选混合分数 0.06963783502578735；底层神经分数 0.06963783502578735。回退原文时所选候选分数为空。

上下文 token：156 → 23；完整输入 token：232 → 99。

| 候选 | 底层神经分数 | 混合分数 | JSON检查通过 | 实际回答正确 |
|---|---:|---:|---|---|
| keep_30 | 0.06963783502578735 | 0.06963783502578735 | True | True |
| keep_60 | 0.292526513338089 | 0.292526513338089 | True | False |

原文关键片段：

```text
The signed data card pairs project Beryl Vela-0024 with Finn Turner in its owner field.
```

主方案所选上下文片段：

```text
The signed data card pairs project Beryl Vela-0024 with Finn Turner in its owner field.
```

观察范围：这些记录确定了本次输入、候选选择和回答的变化；单个案例不能证明模型为何改变回答，也不能证明该规则对所有未来请求都有效。

## 原文答错、主方案仍答错

符合条件的任务数：43；展示 2 例。

### v3-fresh-00029（direct）

问题：What is the current approved budget for project Glade Vela-0029?

Gold：`17044`；原文回答：`13671`；主方案回答：`13671`。

主方案选择 `keep_60`；固定阈值 0.304512918；所选混合分数 0.2659115195274353；底层神经分数 0.2659115195274353。回退原文时所选候选分数为空。

上下文 token：522 → 298；完整输入 token：599 → 375。

| 候选 | 底层神经分数 | 混合分数 | JSON检查通过 | 实际回答正确 |
|---|---:|---:|---|---|
| keep_30 | 0.42167454957962036 | 0.42167454957962036 | True | False |
| keep_60 | 0.2659115195274353 | 0.2659115195274353 | True | False |

原文关键片段：

```text
A check of the completed project file finds approved budget USD 17044 for Glade Vela-0029.
```

主方案所选上下文片段：

```text
The project card for Beryl Peripheral-0029-00 gives owner Clara Morgan, release date 2028-11-24, and approved budget USD 92959.
The project card for Glade Peripheral-0029-01 gives owner Ada Morgan, release date 2028-01-09, and approved budget USD 13671.
A check of the completed project file finds approved budget USD 17044 for Glade Vela-0029.
The project card for Beryl Peripheral-0029-02 gives owner Noah Ross, release date 2028-10-04, and approved budget USD 93801.
The project card for Drift Peripheral-0029-03 gives owner Ada Bennett, release date 2028-09-24, and approved budget USD 85751.
The project card for Lumen Peripheral-0029-04 gives owner Owen Cole, release date 2028-12-26, and approved budget USD 31617.
The project card for Mistral Peripheral-0029-05 gives owner Maya Turner, release date 2028-12-12, and approved budget USD 21802.
```

观察范围：这些记录确定了本次输入、候选选择和回答的变化；单个案例不能证明模型为何改变回答，也不能证明该规则对所有未来请求都有效。

### v3-fresh-00054（header）

问题：What is the current owner for project Glade Vela-0054?

Gold：`Finn Blake`；原文回答：`Finn Grant`；主方案回答：`UNKNOWN`。

主方案选择 `keep_30`；固定阈值 0.304512918；所选混合分数 0.2958596348762512；底层神经分数 0.2958596348762512。回退原文时所选候选分数为空。

上下文 token：516 → 115；完整输入 token：592 → 191。

| 候选 | 底层神经分数 | 混合分数 | JSON检查通过 | 实际回答正确 |
|---|---:|---:|---|---|
| keep_30 | 0.2958596348762512 | 0.2958596348762512 | True | False |
| keep_60 | 0.303827166557312 | 0.303827166557312 | True | False |

原文关键片段：

```text
A project card is headed Glade Vela-0054. On that card, Finn Blake occupies the owner field.
```

主方案所选上下文片段：

```text
The project card for Horizon Peripheral-0054-00 gives owner Finn Brooks, release date 2028-09-09, and approved budget USD 44960.
The project card for Glade Peripheral-0054-03 gives owner Finn Grant, release date 2028-07-26, and approved budget USD 87012.
A project card is headed Glade Vela-0054.
On that card, Finn Blake occupies the owner field.
```

观察范围：这些记录确定了本次输入、候选选择和回答的变化；单个案例不能证明模型为何改变回答，也不能证明该规则对所有未来请求都有效。

## JSON结构检查改变选择

符合条件的任务数：10；展示 2 例。

### v3-fresh-00275（nested_json）

问题：What is the current approved budget for project Horizon Vela-0275?

Gold：`53293`；原文回答：`USD 53293`；主方案回答：`USD 53293`。

主方案选择 `original`；固定阈值 0.304512918；所选混合分数 None；底层神经分数 None。回退原文时所选候选分数为空。

上下文 token：627 → 627；完整输入 token：703 → 703。

| 候选 | 底层神经分数 | 混合分数 | JSON检查通过 | 实际回答正确 |
|---|---:|---:|---|---|
| keep_30 | 0.48732852935791016 | 1.0 | False | False |
| keep_60 | 0.28503212332725525 | 1.0 | False | True |

保持相同主权重及阈值、去掉 JSON 检查时，会选择 `keep_60`，回答 `USD 53293`，实际正确性为 True。该被选候选的原文为合法 JSON，压缩文本不是合法 JSON 容器，因而被结构检查拒绝。

实际纯模型对照 `frozen_v2_mlp` 使用其自身阈值 0.3100292682647705，选择 `keep_60`，回答 `USD 53293`，实际正确性为 True。这个自身阈值与主阈值可能不同。

去掉检查时会选择的候选片段：

```text
{
"certification": {
"operative_catalogue": [
{
"card": {
"operative_attributes": {
"owner": "Mia Morgan",
"approved budget": "USD 89880",
"release date": "2028-11-03"
}
},
"project": "Lagoon Peripheral-0275-00"
},
{
"card": {
"operative_attributes": {
"owner": "Maya Turner",
"approved budget": "USD 53293",
"release date": "2028-02-20"
}
},
"project": "Horizon Vela-0275"
},
{
"card": {
"operative_attributes": {
"owner": "Lena Parker",
"approved budget": "USD 14396",
"release date": "2028-06-26"
}
},
"project": "Mistral Peripheral-0275-01"
},
{
"card": {
"operative_attributes": {
"owner": "Iris Carter",
"approved budget": "USD 47366",
"release date": "2028-01-27"
}
},
"project": "Beryl Peripheral-0275-02"
},
{
"card": {
"operative_attributes": {
"owner": "Clara Carter",
"approved budget": "USD 33975",
"release date": "2028-07-19"
}
},
},
{
"card": {
"operative_attributes": {
"approved budget": "USD 89597",
}
},
"approved budget": "USD 70473",
"approved budget": "USD 27299",
```

原文关键片段：

```text
{
  "card": {
    "operative_attributes": {
      "owner": "Maya Turner",
      "approved budget": "USD 53293",
      "release date": "2028-02-20"
    }
  },
  "project": "Horizon Vela-0275"
}
```

主方案所选上下文片段：

```text
{
  "card": {
    "operative_attributes": {
      "owner": "Maya Turner",
      "approved budget": "USD 53293",
      "release date": "2028-02-20"
    }
  },
  "project": "Horizon Vela-0275"
}
```

观察范围：这些记录确定了本次输入、候选选择和回答的变化；单个案例不能证明模型为何改变回答，也不能证明该规则对所有未来请求都有效。

### v3-fresh-00277（nested_json）

问题：What is the current release date for project Lagoon Vela-0277?

Gold：`2028-03-05`；原文回答：`2028-11-21`；主方案回答：`2028-11-21`。

主方案选择 `original`；固定阈值 0.304512918；所选混合分数 None；底层神经分数 None。回退原文时所选候选分数为空。

上下文 token：624 → 624；完整输入 token：701 → 701。

| 候选 | 底层神经分数 | 混合分数 | JSON检查通过 | 实际回答正确 |
|---|---:|---:|---|---|
| keep_30 | 0.36744585633277893 | 1.0 | False | False |
| keep_60 | 0.26410138607025146 | 1.0 | False | False |

保持相同主权重及阈值、去掉 JSON 检查时，会选择 `keep_60`，回答 `2028-11-21`，实际正确性为 False。该被选候选的原文为合法 JSON，压缩文本不是合法 JSON 容器，因而被结构检查拒绝。

实际纯模型对照 `frozen_v2_mlp` 使用其自身阈值 0.3100292682647705，选择 `keep_60`，回答 `2028-11-21`，实际正确性为 False。这个自身阈值与主阈值可能不同。

去掉检查时会选择的候选片段：

```text
{
"certification": {
"operative_catalogue": [
{
"card": {
"operative_attributes": {
"owner": "Evan Grant",
"approved budget": "USD 77004",
"release date": "2028-04-16"
}
},
"project": "Aquila Peripheral-0277-00"
},
{
"card": {
"operative_attributes": {
"owner": "Leo Reed",
"approved budget": "USD 77373",
"release date": "2028-04-25"
}
},
"project": "Drift Peripheral-0277-01"
},
{
"card": {
"operative_attributes": {
"owner": "Owen Carter",
"approved budget": "USD 42081",
"release date": "2028-03-27"
}
},
"project": "Lumen Peripheral-0277-02"
},
{
"card": {
"operative_attributes": {
"owner": "Leo Parker",
"approved budget": "USD 26930",
"release date": "2028-03-05"
}
},
"project": "Lagoon Vela-0277"
},
{
"card": {
"operative_attributes": {
"owner": "Owen Turner",
"approved budget": "USD 73356",
"release date": "2028-11-21"
}
},
},
{
"release date": "2028-06-04"
}
},
},
"release date": "2028-06-17"
"release date": "2028-03-03"
```

原文关键片段：

```text
{
  "card": {
    "operative_attributes": {
      "owner": "Leo Parker",
      "approved budget": "USD 26930",
      "release date": "2028-03-05"
    }
  },
  "project": "Lagoon Vela-0277"
}
```

主方案所选上下文片段：

```text
{
  "card": {
    "operative_attributes": {
      "owner": "Leo Parker",
      "approved budget": "USD 26930",
      "release date": "2028-03-05"
    }
  },
  "project": "Lagoon Vela-0277"
}
```

观察范围：这些记录确定了本次输入、候选选择和回答的变化；单个案例不能证明模型为何改变回答，也不能证明该规则对所有未来请求都有效。

## 原始证据与限制

本报告只读冻结后的真实结果，未改动模型、阈值、候选回答或标签。风险分数是未校准的排序分数，不能直接解释成保证成立的失败概率。经验校准约束也不能保证未来数据的风险上限。

[完整指标](../results/metrics.json) · [完整案例 JSON](../results/case_studies.json) · [测试前冻结选择](../results/frozen_selection.json)
