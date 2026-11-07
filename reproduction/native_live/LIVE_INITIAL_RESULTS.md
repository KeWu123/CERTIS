# 同Qwen初始证书与实际成本链条（2026-11-07）

在32次原生实时策略执行及独立 checked analysis 全部通过后，执行了两项检查：

- `test_fractional_relaxation.py`：170个独立精确LP顶点枚举实例全部通过，1.170秒。
- `diagnose_live_initial.py`：全部8个query/seed初始状态通过重构和证书校核，14.539秒。

两次退出码均为0，stderr为空，无修源、无失败重跑。完整命令、stdout、用时、执行源SHA
保存在 `initial_checks_execution_01.json`。诊断只读生产raw并在执行完成后分析，没有增加
任何生产策略的计费请求或秒数。

## 主要发现

**8/8个同Qwen初始mask满足 `H_joint <= 0 < H_separate`。**
这是同一具体mask与同一审计证据下的直接证书差异。

**0/8个joint初始discard数超过separate分数松弛上界。**
本轮没有用该上界证出独立于greedy的卡数分离。这不证明两种证书的最优卡数相等，
也不证明joint没有卡数优势；上界在这八个状态上未能排除separate的更优整数计划。
旧70B tape上15/32的卡数分离结论保留在其原有标签和证据域内，不移植到这些Qwen状态。

下表的 `G_j/G_s` 是实际joint/separate greedy初始incident discard数，
`UB_s` 是separate整数最优值的分数松弛上界的floor；`Q`和秒数是实际完整执行成本。
所有初始orphan数和forced discard数均为0。

| Query / key | Seed | G_j / G_s | H_j / H_s on joint mask | UB_s | Q_s → Q_j | Seconds_s → Seconds_j |
|---|---:|---:|---:|---:|---:|---:|
| agent/person, l1 | 101 | 4526 / 4481 | −12233 / 870162 | 4598 | 5486 → 5457 | 285.2673 → 277.3636 |
| agent/person, l1 | 202 | 4549 / 4501 | −4876 / 939068 | 4634 | 5475 → 5445 | 222.2036 → 220.8572 |
| agent/person, l2 | 101 | 3005 / 2647 | −169 / 84486 | 3054 | 6043 → 5844 | 335.5718 → 330.8672 |
| agent/person, l2 | 202 | 2917 / 2623 | −319 / 72393 | 3134 | 6120 → 5895 | 248.8524 → 239.1131 |
| person/athlete, l1 | 101 | 4453 / 4403 | −3901 / 486950 | 4489 | 5558 → 5508 | 310.7946 → 310.7827 |
| person/athlete, l1 | 202 | 4459 / 4407 | −630 / 518320 | 4501 | 5556 → 5506 | 306.7783 → 296.3958 |
| person/athlete, l2 | 101 | 2072 / 1802 | −106 / 79888 | 2116 | 6786 → 6376 | 275.8544 → 267.4148 |
| person/athlete, l2 | 202 | 2107 / 1932 | −199 / 73702 | 2359 | 6634 → 6410 | 378.5110 → 354.4957 |

所有8次匹配中，相对实际separate策略，joint完整请求更少、记录的完整秒数也更少。
合计请求47658→46441（省1217，2.55361%），完整秒数2363.8334→2297.2901
（少66.5432秒，2.81506%）。
这些是4个相关查询各两个审计seed，不是8个独立工作负载；初始mask的证书差异与后续
完整成本差异的联结是描述性的，LP没有证明完整自适应请求或延迟最优性。
person/athlete l1 seed101的时间仅差0.0119秒，应如实保留。

## 可用于解释的同Qwen例子

person/athlete、l2、seed101是本轮完整request gap最大的说明例：

- Joint初始丢2072，separate greedy初始丢1802，greedy差为270。
- 同一个joint mask的 `H_joint=-106`、`H_separate=79888`，只有joint可认证。
- Separate整数最优discard上界为2116，高于2072，**没有据此证明卡数最优值分离**。
- 实际完整请求6786→6376，省410；实际完整秒数275.8544→267.4148，
  少8.4396秒，约3.06%。这些值全部来自相同Qwen cell，不借用70B的支持值或证明下界。

## 数值绑定

- `live_initial_diagnostic.json` SHA256：
  `0071e9d1a6fa859a0baad9d3bbc088e6d2723c819e6fa6ae7a91cb8fdf88e944`
- `live_initial_diagnostic_provenance.json` SHA256：
  `1759dbbde403e8b954ad34e38e605fade44346a026cf9fb9997695e9792d7c92`
- `initial_checks_execution_01.json` SHA256：
  `cd37597683a8d0e12685b975aea85c90cc571bede9b14136f6898b4aad2037ea`
- 执行的诊断源SHA256（与预先静态检查一致）：
  `ccd038e843ed56d131d8be54b3fae3ee3655cee855c2d682ab23d9a6ff9ff17a`
- 执行的170例独立测试源SHA256：
  `6ecd8041f92e13df8ee0f3c062c537bcc72b8f90b7a7a367b81ba53934164547`

输出JSON保留每个初始audit事实、mask、key/side计数、精确分数上界、实际成本、32份
checked raw及16份用于证书对照raw的SHA，并绑定生产源和诊断源。生产源、已发布v3
与科学raw均未改动。
