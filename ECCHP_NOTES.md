# ECCHP 工程实现记录（按 `reference/5.md` 路线 A）

基线提交：`babb2b7b72ce5cfc15d3aa42e3d98c5ff412c2de`
（分支 `research/exec-aware-hwm`，工作区在其之上叠加本次改动）

本文件对应研究计划 **阶段 0–2**。阶段 3 之后（plan switcher / execution
calibrator / 长期价值回归）尚未实现，不在本次范围内。

---

## 1. 阶段 0：先修复会污染新实验的问题

| 问题 | 处理 |
| --- | --- |
| `plan_with_high_level_reselection` 不接受 `min_improvement`（调用方会 `TypeError`，又被外层兜底吞掉） | 签名与调用方对齐，支持标量或 `(B,)` 张量边距 |
| E4 注释声称"联合评分"，但接受条件仍强制要求低层 gap 改善 | 新增 `repair_require_gap_improvement`（默认 `true` 保持历史行为；ECCHP 运行时置 `false`），接受条件只保留可行域 + 组合分数；每次决策记录 `gap_improved` / `acceptance_rule` 便于对比 |
| 候选全部被拒后仍消耗全局随机流 | 新增 `repair_restore_rng_on_reject`（默认 `true`）：无任何环境接受时恢复决策前的 CPU/CUDA RNG |
| 影子模式缺失 | reselection 分支支持 `repair_force_reject`（与投影分支一致），用于"只算候选、不改轨迹"的验证 |
| 候选生成异常 | 正常模式记录 `repair_error` 事件并安全回退；`repair_raise_errors=true` 时抛出 |

对应测试：`tests/test_planner_contract.py`

* 关闭全部开关 ⇒ 动作序列可复现；
* 影子模式 / 仅收集候选 ⇒ 与 Warm-start Only 的动作序列**逐元素相同**；
* `B = 1, 2, 5` 每个环境只用自己的状态、候选与历史；
* 候选生成异常 ⇒ 正常模式安全回退、`repair_raise_errors=true` 时抛出；
* 被选中的候选与实际执行的 L1 计划一致（动作编码可验证），
  且 L1/L2 的 warm-start 缓存与选择结果一致；
* `PlanningResult.actions` 已是**动作空间**动作（`MPPIPlanner.plan` 末尾归一化后），
  回放时不能改用 MPPI 内部的 `ctrl.U`（搜索空间）。

另外修正了工作区里会导致整个分层评估无法启动的语法错误（`d4rl/mpc.py` 中
`trials_pa th=` → `trials_path=`）。

---

## 2. 阶段 1：计划级诊断

新增模块：

* `pldm/planning/plan_memory.py`
  * `PlanMemory`：**逐环境**的 `plan_id` / `decision_id` / `plan_age`；
    关键决策只"老化"计划，替换才分配新 id；
  * `PlanDecision` 数据类（`plan_decision` 事件）。
* `pldm/planning/progress_estimator.py`
  * `RouteHistory`：基于**真实 XY** 的滑窗指标
    `route_path_length`（$L_t$）、`route_efficiency`（$R_t$）、
    `route_revisit_count`。
* `pldm/planning/diagnostics.py`
  * `ExecutionOutcomeRecord`、`PlanTransitionRecord`；
  * 记录器方法 `record_plan_decision / record_execution_outcome / record_plan_transition`；
  * `enabled` 现在也认 `log_plan_events` 与 `collect_candidates`。

接入点：

* `TwoLvlPlanner.plan(..., decision_step=...)`：每次重规划生成 `last_plan_decision`
  （`decision_id`、`plan_id`、来源、切换原因、候选数、`J_H`、`cross_mse`、
  `pred_progress_h`），逐环境独立；
* `MPCEvaluator._perform_mpc`：写入 `plan_decision` / `plan_transition`，
  在下一个重规划窗口写入 `execution_outcome`（真实位移、子目标误差、
  `execution_progress_gap`、路线指标、`stuck`），三个事件通过
  `decision_id` + `plan_id` 关联；
* 打开方式：`exec_repair.log_plan_events=true`（`disabled` 时零开销）。

注意：这些指标只是诊断信号，不能直接当失败判据（绕障本身效率就低）。

---

## 3. 阶段 2：反事实候选（研究关口 G1）

新增：

* `pldm/planning/env_snapshot.py`：dm_control `physics.get_state/set_state`
  或 gym `sim.get_state/set_state` 的完整状态快照 + RNG 状态；
  `verify_snapshot_roundtrip` 用于收集前的一次性校验；
* `pldm/planning/maze_metrics.py`：`graph_distance` / `graph_progress`
  （仅离线诊断，基于 `NavigationWrapper.find_shortest_path`）；
* `pldm/planning/candidate_evaluator.py`：
  Oracle Candidate Gain、Kendall τ-b、pairwise accuracy、
  best-candidate hit rate、模型选择的 regret、bootstrap CI；
* `pldm/planning/subgoal_repair.py`：`RepairInfo.candidate_evidence`
  暴露每个候选的 `J_H` / `C_L` / 预测进展 / `PlanningResult.actions`（仅内存）；
* `MPCEvaluator._collect_counterfactual_returns`：
  `collect_candidates=true` 时，对每个触发决策
  ① 快照全部环境与 planner warm start；
  ② 逐候选从**同一真实状态**执行候选动作块（首个 chunk）；
  ③ 之后继续使用**同一个 HWM 控制器**（统一后续策略）至 40/80 步；
  ④ 记录 `candidate_outcome`（真实图距离收益 `real_return`）；
  ⑤ `fork_rng` + 快照恢复，不改变事实轨迹。

脚本：

```bash
# 1) 收集（配置里打开 exec_repair.collect_candidates=true 跑评估）后整理候选表
python scripts/collect_plan_candidates.py \
    --diagnostics checkpoint/.../run_dir \
    --out results/candidates_task7

# 2) 研究关口 G1 统计
python scripts/evaluate_candidate_oracle.py \
    --candidates results/candidates_task7_candidates.jsonl \
    --horizons 40 80 \
    --task-groups task_groups.json \
    --out results/oracle_task7

# 无实验数据时先验证管线
python scripts/evaluate_candidate_oracle.py --demo
```

候选动作回放使用 `PlanningResult.actions`，与 `_perform_mpc` 里
`envs[j].step(planned_actions[j, 0])` 同一空间，避免 `ctrl.U` 的搜索空间问题。

---

## 4. 新增 / 变更的配置项（`ExecRepairConfig`）

```yaml
exec_repair:
  # 阶段 0
  repair_require_gap_improvement: true   # false = ECCHP 接受准则（组合分数）
  repair_restore_rng_on_reject: true
  # 阶段 1
  log_plan_events: false
  route_history_window: 8
  revisit_radius: 0.25
  # 阶段 2
  collect_candidates: false
  candidate_horizons: "40,80"
  candidate_max_decisions: 0             # 0 = 不限制
```

所有开关默认保持旧行为；`enabled=false` 时新增代码路径完全不进入。

---

## 5. 测试

```bash
cd /root/autodl-tmp/.autodl
python -m unittest discover -s tests -v
```

当前 45 个用例，覆盖：

* `test_planner_contract.py`：阶段 0 的状态/随机流隔离契约；
* `test_plan_memory.py`、`test_progress_and_snapshot.py`：计划身份与路线指标；
* `test_diagnostics_events.py`：三类事件字段与关联；
* `test_candidate_evaluator.py`：G1 统计量；
* `test_scripts_smoke.py`：收集 + oracle 脚本端到端。

---

## 6. 已知限制 / 下一步

1. 候选集合目前只在 **repair 触发**的决策上产生（沿用 E4 触发逻辑），
   因此 G1 统计的是"规划器自认为可疑"的决策分布，报告时需注明这一条件；
2. 反事实回放：首个 chunk 用候选动作，之后用 HWM 统一后续策略；
   回放成本 = 候选数 × 时域 × 40/80 步，建议先用 `candidate_max_decisions` 限流；
3. `candidate_evidence` 只存在内存中，未落盘为张量文件（计划中建议的
   `.pt` 快照）；若要跨进程复算，可在此基础上加 `torch.save`；
4. 阶段 3–9（Plan Persistence 在线决策、Execution-Calibrated Value、
   Counterfactual Switching、校准网络）尚未实现。
