"""客流报表约束求解核心（纯函数，不依赖数据库 / Web 框架）。

建模
----
每份报表 r = (编号, 首天 s, 末日 e, 总人次 t) 表示

    sum_{d in [s, e]} x_d = t,   x_d >= 0 且为整数

引入前缀和 P_d = sum_{k <= d} x_k（锚点 P_0 = 0），报表约束变为

    P_e - P_{s-1} = t
    P_d - P_{d-1} >= 0                 （日人次非负）

这是差分约束系统。对 P_v - P_u >= w 建图 u -> v，边权 w：

* 报表等式拆成两条边：(s-1) -> e，权 t；e -> (s-1)，权 -t
* 相邻节点间补权 0 的链边（非负性）

为控制规模，图中只放报表端点（s-1、e），节点数 <= 241。

可行性
~~~~~~
系统可行 <=> 图中没有正权环。Floyd-Warshall 求全源最长路 L，
若存在 L[i][i] > 0 即不可行；诊断时用 Bellman-Ford 找一条正权环作为
不可行的“证据”。

逐日唯一值
~~~~~~~~~~
设相邻两个图节点 a < b 之间（空档 (a, b]）没有任何报表端点，则该空档内
每天的人次只受“空档总量 P_b - P_a 非负、可在各日间任意拆分”约束：

* 空档长度为 1 天：该日人次就是 P_b - P_a，
  最小值 L[a][b]，最大值 -L[b][a]（最短路 = 反向最长路取负）；
* 空档长度 >= 2 天：任一单日的最小值为 0（流量可放在别日），
  最大值仍为 -L[b][a]（单日可吸走整个空档的流量）。

最小值 = 最大值（且有限）时，该日人次被唯一确定；整数边权保证结果为
整数。日期不在任何报表区间内时没有有限界，返回“待补”。

冲突组
~~~~~~
Bellman-Ford 检测到正权环时，环上的报表本身就已经不能同时成立：把环上
的“报表边”换成 0 权链边上的等价路径后，环仍然存在。因此只需在环上这些
报表中做删除法求最小冲突组（IIS），不必拿整批报表逐份试删；整个过程只
判断可行性，不算逐日人次。
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Iterable, Sequence

import numpy as np

MAX_SPAN_DAYS = 90
MAX_REPORT_COUNT = 120
# 现实护栏：90 天景区总人次不可能超过 10^12，同时保证 int64 路径和不溢出。
MAX_TOTAL = 10**12
_NEG = -(10**15)


class ReportValidationError(ValueError):
    """单份报表自身不合法（跨期超 90 天、负数等）。"""


@dataclass(frozen=True)
class Report:
    report_id: str
    start_date: date
    end_date: date
    total: int

    def validate(self) -> None:
        if not self.report_id or not self.report_id.strip():
            raise ReportValidationError("报表编号不能为空")
        if self.end_date < self.start_date:
            raise ReportValidationError(
                f"报表 {self.report_id} 的末日 {self.end_date} 早于首天 {self.start_date}"
            )
        span = (self.end_date - self.start_date).days + 1
        if span > MAX_SPAN_DAYS:
            raise ReportValidationError(
                f"报表 {self.report_id} 档期 {span} 天，超过最长 {MAX_SPAN_DAYS} 天"
            )
        if not isinstance(self.total, int) or isinstance(self.total, bool):
            raise ReportValidationError(f"报表 {self.report_id} 的总人次必须是整数")
        if self.total < 0:
            raise ReportValidationError(f"报表 {self.report_id} 的总人次不能为负")
        if self.total > MAX_TOTAL:
            raise ReportValidationError(
                f"报表 {self.report_id} 的总人次 {self.total} 超过上限 {MAX_TOTAL}"
            )


@dataclass
class SolvedSystem:
    """一套生效报表的求解结果。"""

    feasible: bool
    reports: tuple[Report, ...]
    # 覆盖日期 -> 唯一次数；未出现的日期即未覆盖（待补）
    determined: dict[date, int]
    # 覆盖但无法唯一确定的日期
    covered_undetermined: frozenset[date]


@dataclass(frozen=True)
class _Graph:
    """差分约束图：节点 0 为锚点，其余节点对应报表端点日期。"""

    reports: tuple[Report, ...]
    node_dates: list[date]
    n: int
    # (u, v, 权, 产生该边的报表编号)；链边没有报表编号。
    edges: list[tuple[int, int, int, str | None]]


def _build_graph(reports: Iterable[Report]) -> _Graph:
    """校验并按“同编号后者覆盖前者”的语义建图（只含报表端点）。"""
    unique: dict[str, Report] = {}
    for r in reports:
        r.validate()
        unique[r.report_id] = r  # 同编号以后者为准（整份替换语义）
    ordered = tuple(unique.values())
    if not ordered:
        return _Graph((), [], 0, [])

    boundary: set[date] = set()
    for r in ordered:
        boundary.add(r.start_date - timedelta(days=1))
        boundary.add(r.end_date)
    node_dates = sorted(boundary)
    idx = {d: p + 1 for p, d in enumerate(node_dates)}
    n = len(node_dates) + 1

    edges: list[tuple[int, int, int, str | None]] = [
        (i - 1, i, 0, None) for i in range(1, n)  # 非负链边
    ]
    for r in ordered:
        s_node = idx[r.start_date - timedelta(days=1)]
        e_node = idx[r.end_date]
        edges.append((s_node, e_node, r.total, r.report_id))
        edges.append((e_node, s_node, -r.total, r.report_id))
    return _Graph(ordered, node_dates, n, edges)


def _floyd_longest(node_count: int, edges: Sequence[tuple[int, int, int]]) -> np.ndarray:
    """全源最长路。不可达为 _NEG；正权环表现为对角线 > 0。"""
    mat = np.full((node_count, node_count), _NEG, dtype=np.int64)
    np.fill_diagonal(mat, 0)
    for u, v, w in edges:
        if w > mat[u, v]:
            mat[u, v] = w
    for k in range(node_count):
        mat = np.maximum(mat, mat[:, k, None] + mat[None, k, :])
    return mat


def _positive_cycle_reports(graph: _Graph) -> set[str] | None:
    """Bellman-Ford 找正权环；可行返回 None，否则返回环上涉及的报表编号。

    所有节点初值 0（等价于挂一个到所有节点的 0 权超级源），因此任何正权环
    都会在第 n 轮松弛中继续更新。沿前驱回退 n 步即可落进环内，再沿前驱
    收集环上的边，取出报表编号。
    """
    if graph.n == 0:
        return None
    n = graph.n
    dist = [0] * n
    pred_node = [-1] * n
    pred_edge = [-1] * n
    last_changed = -1

    for _ in range(n):
        changed = False
        for ei, (u, v, w, _rid) in enumerate(graph.edges):
            candidate = dist[u] + w
            if candidate > dist[v]:
                dist[v] = candidate
                pred_node[v] = u
                pred_edge[v] = ei
                changed = True
                last_changed = v
        if not changed:
            return None  # n 轮内收敛 => 无正权环

    # 第 n 轮仍在更新 => 存在正权环；回退 n 步保证落在环上。
    y = last_changed
    for _ in range(n):
        y = pred_node[y]
        if y < 0:
            return set()  # 兜底：环提取失败，交由调用方全池诊断
    ids: set[str] = set()
    cur = y
    for _ in range(n):
        rid = graph.edges[pred_edge[cur]][3]
        if rid is not None:
            ids.add(rid)
        cur = pred_node[cur]
        if cur == y:
            break
    else:  # pragma: no cover - 理论上 y 必在环上，这里只防御异常前驱链
        return set()
    return ids


def is_feasible(reports: Iterable[Report]) -> bool:
    """只判断报表能否同时成立，不计算逐日人次（用于补录验算与冲突诊断）。"""
    graph = _build_graph(reports)
    if not graph.reports:
        return True
    return _positive_cycle_reports(graph) is None


def solve(reports: Iterable[Report]) -> SolvedSystem:
    """对一整套报表求解可行性与每日唯一值。"""
    graph = _build_graph(reports)
    ordered = graph.reports
    if not ordered:
        return SolvedSystem(True, (), {}, frozenset())

    covered: set[date] = set()
    for r in ordered:
        cur = r.start_date
        while cur <= r.end_date:
            covered.add(cur)
            cur += timedelta(days=1)

    longest = _floyd_longest(
        graph.n, [(u, v, w) for u, v, w, _rid in graph.edges]
    )
    if np.any(np.diag(longest) > 0):
        return SolvedSystem(False, ordered, {}, frozenset(covered))

    # --- 逐日上下界 ------------------------------------------------------
    node_dates = graph.node_dates
    # node_dates[i] 的矩阵索引为 i+1；锚点（索引 0）对应日期为其前一天。
    node_date_with_anchor = [node_dates[0] - timedelta(days=1)] + node_dates
    determined: dict[date, int] = {}
    undetermined: set[date] = set()
    for d in covered:
        p = bisect_left(node_dates, d)  # 第一个 >= d 的节点日期（0 基位置）
        if p < len(node_dates) and node_dates[p] == d:
            lo = p          # 左侧节点矩阵索引 = p+1-1 = p
            hi = p + 1      # d 本身是节点
        else:
            lo = p          # 左侧节点 = node_dates[p-1]，矩阵索引 p
            hi = p + 1      # 右侧节点 = node_dates[p]，矩阵索引 p+1
        gap_days = (node_date_with_anchor[hi] - node_date_with_anchor[lo]).days
        x_min = int(longest[lo, hi]) if gap_days == 1 else 0
        x_max = int(-longest[hi, lo])
        if x_min == x_max:
            determined[d] = x_min
        else:
            undetermined.add(d)

    return SolvedSystem(
        feasible=True,
        reports=ordered,
        determined=determined,
        covered_undetermined=frozenset(undetermined),
    )


def find_minimal_conflict(
    reports: Sequence[Report],
    keep_ids: set[str] | None = None,
) -> list[str]:
    """在不可行的报表集合中求一个最小冲突组（删除型 IIS）。

    先取正权环上的报表作为候选（它们自身已不可行），只在候选内做删除法，
    避免拿整批报表逐份试删；所有判断都只做可行性，不计算逐日人次。

    keep_ids 中的报表在删除法中靠后尝试，使冲突组更倾向落在这些报表上
    （补录场景下即优先指出本批中无法成立的报表），不影响最小性。

    最小性：去掉组内任意一份，组内其余报表即可同时成立。
    """
    graph = _build_graph(reports)
    if not graph.reports:
        return []
    cycle_ids = _positive_cycle_reports(graph)
    if cycle_ids is None:
        return []

    pool = list(graph.reports)
    if cycle_ids:
        by_id = {r.report_id: r for r in pool}
        pool = [by_id[rid] for rid in cycle_ids]
        if is_feasible(pool):
            # 理论上正权环上的报表必然互斥；兜底退回整池删除法。
            pool = list(graph.reports)

    # 删除法：逐个检查删掉后是否仍不可行；仍不可行说明它不是必要成员。
    keep = keep_ids or set()
    ordered = sorted(pool, key=lambda x: (x.report_id in keep, x.report_id))
    kept = list(ordered)
    for r in ordered:
        rest = [x for x in kept if x.report_id != r.report_id]
        if rest and not is_feasible(rest):
            kept = rest
    return sorted(r.report_id for r in kept)
