# Native live 成本审计（2026-11-07）

本文件只审计现有 v3 成本口径并建议最小补充矩阵；没有运行新推理，没有改写历史结果。

## 1. 旧 24 次实时闭环实际测量了什么

核查来源：`paper/certis_icde2027.tex` 的 Live closed-loop cost 段、`reproduction/completion/live_closed_loop.py`、`analyze_completion.py`、`native_dbpedia.py`，以及它们调用的 `study/run_study.py`、`study/cached_greedy.py`、`runtime/external_study/common.py`。

计费总时间为：

```
complete_seconds = algorithm_seconds
                 + sql_preprocessing_seconds
                 + fact_ingest_seconds
                 + returned_materialization_seconds
```

- `algorithm_seconds` 包含付费 audit/execution 的真实 `/completion` HTTP 往返 wall time、审计区间反演、排序/规划、事实更新、伙伴剪枝和账本整理。实时返回标签直接进入该策略的私有 facts。
- joint 的查询局部缓存建立、清理、动态模块加载等 wrapper 开销也计入 `algorithm_seconds`。其 planner 子计时是诊断分项，不能再次加到总时间中。
- SQL preprocessing 实际包含新建 DuckDB 连接、incidence/nodes 输入和临时表；facts ingest 和返回 bag 的 SQL 物化计费。
- 排除模型加载、health wait、warmup、稳定性预检、完整本地参考标签获取、reference_labels SQL 加载、full_derivations SQL 建表及 `EXCEPT ALL` 独立核验。保存的 `outer_seconds` 包含运行内的参考和核验开销；它不是论文的 `complete_seconds`。
- 每次实时 acquisition 的逐标签 reference 一致性检查发生在算法 timer 内，因此这点检查开销实际仍计费；最终全量 SQL/reference 检查单独排除。`outer_seconds - complete_seconds` 是排除项及 wrapper 杂项的合计，不应全称为纯 SQL 核验时间。
- 每个策略执行前调用 `case.close_sql()`，从而每次计费 SQL setup；每个策略新建空 oracle facts。模型服务持续 warm，无 prompt/KV 复用。这里是“模型 warm、应用事实和 SQL 连接 cold”，不是完整机器冷启动。

旧汇总值可从 `results/analysis.json` 复核：exact 773.129630 s / 23,490 requests；per-side 792.410912 s / 24,534；joint greedy 783.065873 s / 24,320。joint 比 exact 慢 1.2852%，比 per-side 快 1.1793%；并且比 exact 多 3.5334% 请求。表中的四行两次均值之和是上述总数的一半。每种方法只有四个 query、每 query 两遍；approval/disapproval-thread 的 exact 时间范围为 76.80–96.10 s，不能据两遍给出显著性结论。

## 2. 单槽推理的结论边界

旧成功实验使用 `--parallel 1 --no-cont-batching`，逻辑 acquisition batch 最大为 128。同一个逻辑 batch 内的 GPU 推理串行；逻辑 batch 数量不能等同于 GPU 并行批次数，也不能把 batch 少直接解释为批处理吞吐提升。

每个 batch 保存真实 HTTP wall time 和各响应 token/timing；计费应保留这一实测口径，不能用离线 replay 时间或常数 latency 替代。已有 128-slot fixed-trace 结果是另一成本情景，不能与该闭环汇总。

旧预检对固定 128 个 prompt 使用逆序和 cap 1/8/128；所有付费 live 标签再对独立完整 Qwen 参考检查。这支持观察输入及观察调用次序的一致性，不是 runtime 的普遍确定性证明。128-slot 失败已保留；新实验若更改并发、prompt、截断规则、context 或模型，必须使用新的 oracle binding、重新取参考并预检。若发生 mismatch，保存失败并停止该 binding，不能代入参考标签继续。

## 3. 新比较必须固定的条件

1. **查询与输出**：同一 row roster、原生 l1/l2 incidence、谓词版本、相同 bag SQL，保留 same-article pairs；类别元数据不充当付费 oracle 标签。各策略从空 identity facts 开始，reference 只供核验。
2. **质量目标**：近似方法均为 bag recall 0.9、同一 delta=0.05、同预算和 audit seed；per-side 每批连续重算 split。per-side 的逐 key 约束更强，应准确称为“同一最终 bag 目标下的 per-side baseline”，不能称与全局法完全相同的可行域。
3. **认证增量**：global separate 与 joint 固定相同局部候选空间、greedy allocator、排序、tie-breaking、不可撤销 discard、正例保留、伙伴剪枝、batch cap 与 query-local memoization；只改变 support/certificate。DP 不必加入最小 live 矩阵。
4. **执行方向**：exact right-first 在协议中固定；不根据事后成本选择更便宜 exact 方向。近似方法使用相同 degree order。不同策略允许因其合法决策产生不同实际请求顺序，须逐 batch 保存。不要强制事后相同 trace 来取代自适应实验。
5. **时序与缓存**：每个 run 清空 facts/SQL/profile caches；同一 warm 模型、相同 prompt caching 关闭条件，固定暖机后轮转方法顺序。保存实际运行顺序和时间戳，不能仅保存 seed。
6. **实现版本**：新旧 profile builder 选一个后在新矩阵中固定。若另做 builder A/B，应独立成对测量，不把微基准时间扣减旧闭环。separate 和 joint 的缓存资源上限/重置策略应对称，setup/cleanup 均计费。

现有 `common.evaluate` 先构建完整 reference SQL，再计 returned SQL；虽然参考建表时间排除，仍可能改变内存/CPU 状态。最小补充可沿用且向全部策略一致应用，并如实说明。若新 driver 解耦，应先计费 production SQL/返回输出，再做独立 reference 构建，且所有策略同步采用，避免悄悄改变一方的成本边界。

## 4. 必须保存的字段

- Manifest：协议和全套依赖源代码 hash；数据/row selection/target hash；model shards、server binary/build hash；完整 server command、`/props`；prompt 模板/谓词问句和输入规范化/截断规则；decoding/grammar/seed/context；slot、continuous batching、logical cap、cache 开关；硬件/软件版本。
- Run：query/pair/key/target、method/backend、audit budget/seed、repetition、实际方法顺序、开始结束时间、initial facts count、cache reset 和 SQL scope；成功/失败/coverage conflict 状态。
- Paid ledger：predicate+row identity、phase、请求顺序、label、input/output tokens；每个 batch 的 ids/size、真实 HTTP wall、原始响应及 runtime timing，所有 parse/error/retry 及其已付费成本。
- Cost：audit/execution 各自 requests/tokens/logical batches/inference wall；planner wall/CPU、audit inversion、bookkeeping、cache setup/cleanup、SQL setup/facts ingest/returned SQL；complete、outer 与明确列出的排除项。分项重叠关系须明示，避免重复相加。
- Correctness：完整参考 binding/hash、逐 acquisition mismatch 检查；regions/intervals、facts、初始和最终 discard、规划事件/support、full/returned/lost bag counts、`EXCEPT ALL` subset 检查、coverage 和 conditional loss 检查。新模型结果与旧 70B tape 分开。
- 机制衔接：separate/joint 的同 snapshot、同 mask support 和选中 mask/support；证书松紧、初始 discard、最终 paid requests、总 wall 的变化应能追溯。对 planner event 的 pooled 分布不能冒充独立 query 数量。

## 5. 建议最小矩阵及统计

在看新 Qwen 输出前固定既有谓词顺序的前两对 `agent/person`、`person/athlete`；每对都保留 l1 和 l2，使用全部 5,000 articles/side。四策略为 exact right-first、adaptive continuous per-side、global separate+greedy、joint+greedy；audit 500/side、seeds 101/202，轮转顺序，共 **4 query × 4 policy × 2 pass = 32 次真实闭环**。只有三个 unique predicates，完整参考可共享为 15,000 次独立 acquisition；参考事实不注入策略缓存。无需同时扩展八对谓词、两个 audit budgets 和 DP。

时间控制应来自这个事先固定的小矩阵及统一输入长度上限；若必须使用 hash 子集，选择规则和规模须在查看新标签前冻结，并明确它改变了 key 大小和收益机会。不得根据新 Qwen 结果挑正收益 pair 或只保留 l1。若设置时限，按事先固定顺序停止，保留未完成/失败，不把幸存成功 cell 汇总成完整矩阵。

每个 query/policy 先对两遍给 mean 和 min–max，同时公开每遍请求数及成本。分别按 l1/l2 报 pooled saving `1-sum(C_g)/sum(C_ref)`；再报告 query 等权宏平均 `mean_q(1-mean_r(C_gqr)/mean_r(C_refqr))`、median、最差值以及 query-level W/T/L。seed-level W/T/L 可另列并注明 matched runs。重复 seeds 是同一 query 的审计/时间变化，不能写为更多独立 workloads；同一谓词对的 l1/l2 亦是关联的结构对照。仅四个 queries、两个 predicate pairs 的矩阵支持案例和机制证据，不支持总体显著性或广泛提速保证。

若最终 joint 在粗 key 也未胜 exact，应保留该结果，把完整成本边界讲清楚；不能将新的 planner 微基准加速或旧 70B 请求节省换算成尚未实测的实时端到端收益。

## 6. 新 native SQL 测量接口与历史口径差异

新增 `cost_sql.py::returned_first_evaluate(case, facts)`，保留 `common.evaluate` 的全部输出字段和相同返回/参考 bag SQL。它先建立 incidence/nodes，写入本次付费 facts，物化 `returned_derivations`，再访问参考标签、构建 `full_derivations` 并运行 `EXCEPT ALL`。因此新 native 冷查询的返回 SQL 不再由先前 reference SQL 建表暖内存。

计费总式保持三项 SQL 时间相加；参考标签加载、参考 bag 建表、参考 count、事实一致性检查及 bag 核验均独立记录并排除。`reference_count_seconds`、`acquired_facts_reference_check_seconds` 是新增的排除项诊断；`sql_validation_seconds` 包含返回 count、SQL 事实一致性及两向 `EXCEPT ALL`。`sql_evaluation_order` 和 `cold_returned_first` 可验证所有策略确实调用同一路径。

缓存调用仍与 `Case._sql` / `Case.close_sql()` 兼容：重复调用只复用 metadata 和固定 reference，重新写入付费 facts、物化返回 bag 和检查返回 count；单次 setup/reference 构造时间为零，新的 facts ingest/returned CTAS 仍计费。正式比较必须在每次 policy 前 `close_sql()`，确保 `cold_returned_first=True`。若诊断时复用旧 `common.evaluate` 状态，明确标记 `cached-reference`，不能称冷测量。

独立 CPU 小 fixture 测试位于 `test_cost_sql.py`；2026-11-07 使用本地研究 Python 运行，8 项测试全部通过（0.322 s）。覆盖 exact/subset/empty 与旧 evaluator 的所有非时间字段一致、空参考答案、多 key 的同一 pair 重复 derivation、cached SQL 重新计数和计费规则、参考读取发生在 returned CTAS 之后、独立 parquet 参考不注入付费 facts、错误 paid label 拒绝及连接清理、旧缓存状态兼容。

这是新 native live 测量口径的显式调整。旧 GoEmotions 24 次闭环结果仍保持原始 evaluator 和原始时间；不得使用新 evaluator 重算历史时间后与论文旧表混用。

## 7. 运行中的只读缓存核查

2026-11-07 17:19 只读检查确认，facts、SQL连接和profile memo按既定协议逐run重置；exact_hypergeom.count_interval_exact与其分母组合数使用进程级LRU，未逐run清空。因此“冷”只描述facts/SQL/profile，而非全部算术库状态。各方法仍按固定rotation顺序计实际audit开销，不以旧微基准扣减。

第一对谓词已完成8个run的独立成本记录中，raw_audit_beyond_inference为0.0086–0.1571秒（exact为0），该字段仍含audit阶段inline monitor。明显的audit wall波动主要在实测inference部分，不能归因于区间反演。完整32run的生产计时保持原样；该核查没有修改运行源、缓存或model配置。
