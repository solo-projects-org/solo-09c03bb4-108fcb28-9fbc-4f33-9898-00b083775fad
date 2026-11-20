"""退回处理优化：缩小冲突排查范围、只判可行性、快照式冲突组、诊断期间可放行其它批次。"""

import threading
import time
from datetime import date, timedelta

import pytest

from app import solver
from app.database import ReportRow, SessionLocal
from app.main import service
from app.service import BatchConflictError, ReportService
from app.solver import Report, find_minimal_conflict, is_feasible, solve

D = date(2024, 3, 1)


def d(offset: int) -> date:
    return D + timedelta(days=offset)


def r(rid: str, s: int, e: int, total: int) -> Report:
    return Report(rid, d(s), d(e), total)


def _seed(reports):
    with SessionLocal() as s:
        service.upload_batch(s, reports)


def _stored_totals() -> dict[str, int]:
    with SessionLocal() as s:
        return {row.report_id: row.total for row in s.query(ReportRow).all()}


# --------------------------------------------------------------------------- #
# 只判可行性、不逐日计算
# --------------------------------------------------------------------------- #
def test_is_feasible_matches_solve_on_random_cases():
    import random

    rng = random.Random(1234)
    for _ in range(120):
        reports = []
        for k in range(rng.randint(1, 7)):
            s = rng.randint(0, 10)
            reports.append(
                r(f"R{k}", s, rng.randint(s, min(10, s + 5)), rng.randint(0, 6))
            )
        assert is_feasible(reports) == solve(reports).feasible


def test_conflict_diagnosis_never_computes_daily_values(monkeypatch):
    def _boom(*_args, **_kwargs):
        raise AssertionError("诊断冲突组时不应计算逐日人次（solve）")

    monkeypatch.setattr(solver, "solve", _boom)
    reports = [r("A", 0, 2, 5), r("B", 0, 2, 6), r("C", 3, 4, 2)]
    assert find_minimal_conflict(reports) == ["A", "B"]


# --------------------------------------------------------------------------- #
# 缩小排查范围
# --------------------------------------------------------------------------- #
def test_diagnosis_only_tries_reports_on_positive_cycle(monkeypatch):
    """120 份独立报表中只有 2 份冲突：不应拿全批逐份试删。"""
    reports = [r(f"G{i:03d}", i * 100, i * 100 + 59, 1000) for i in range(120)]
    reports += [r("A", 50, 51, 5), r("B", 50, 51, 7)]
    calls: list[int] = []
    real = solver.is_feasible

    def counting(rs):
        rs = list(rs)
        calls.append(len(rs))
        return real(rs)

    monkeypatch.setattr(solver, "is_feasible", counting)
    group = find_minimal_conflict(reports)
    assert set(group) == {"A", "B"}
    # 可行性判断次数与“环上报表数”同阶，而不是全批 122 份。
    assert len(calls) <= 10, calls
    assert max(calls) <= 5, calls


def test_returned_group_is_minimal_on_large_set():
    reports = [r(f"G{i:03d}", i * 100, i * 100 + 59, 1000) for i in range(60)]
    reports += [r("A", 50, 51, 5), r("B", 50, 51, 7), r("C", 50, 51, 9)]
    group = find_minimal_conflict(reports)
    by_id = {x.report_id: x for x in reports}
    grp = [by_id[g] for g in group]
    assert not solve(grp).feasible
    for member in group:
        assert solve([x for x in grp if x.report_id != member]).feasible


# --------------------------------------------------------------------------- #
# 服务层：整批退回、快照式冲突组、诊断期间放行其它批次
# --------------------------------------------------------------------------- #
def test_rejected_batch_is_stored_atomically():
    _seed([r("A", 0, 2, 5)])
    before = _stored_totals()
    with SessionLocal() as s:
        with pytest.raises(BatchConflictError) as exc:
            service.upload_batch(
                s,
                [
                    r("A", 0, 2, 6),  # 替换 A
                    r("B", 0, 2, 7),  # 与 A 矛盾
                    r("C", 50, 50, 1),  # 与冲突无关
                ],
            )
    assert "A" in exc.value.conflict_ids and "B" in exc.value.conflict_ids
    assert "C" not in exc.value.conflict_ids
    assert _stored_totals() == before  # 整批不入库，A 也未替换


def test_conflict_group_uses_snapshot_taken_when_batch_was_validated(monkeypatch):
    """诊断期间同号报表被更正成与补录一致，退回结果仍按验算当时的报表给出。"""
    _seed([r("A", 0, 2, 5)])
    captured: dict = {}
    real = solver.find_minimal_conflict

    def diagnose_then_correct(reports, keep_ids=None):
        captured["snapshot"] = sorted((x.report_id, x.total) for x in reports)
        # 模拟另一笔补录在诊断期间把同号 A 更正为 9：若重新读库，本批就自洽了。
        with SessionLocal() as s:
            service.upload_batch(s, [r("A", 0, 2, 9)])
        return real(reports, keep_ids=keep_ids)

    monkeypatch.setattr(solver, "find_minimal_conflict", diagnose_then_correct)
    with SessionLocal() as s:
        with pytest.raises(BatchConflictError) as exc:
            service.upload_batch(s, [r("X", 0, 2, 9)])

    # 冲突组来自验算当时快照里的 A=5、X=9，而不是诊断时库里已被更正成 9 的 A。
    assert ("A", 5) in captured["snapshot"]
    assert set(exc.value.conflict_ids) == {"A", "X"}
    # 更正已入库，被退回的 X 整批没有入库。
    assert _stored_totals() == {"A": 9}


def test_other_batch_commits_while_conflict_group_is_being_located(monkeypatch):
    """判定不能入库后，定位冲突组期间其它通过验算的批次可以先入库。"""
    _seed([r("A", 0, 2, 5)])
    entered = threading.Event()
    release = threading.Event()
    real = solver.find_minimal_conflict

    def blocking(reports, keep_ids=None):
        entered.set()
        assert release.wait(10), "主测试线程未放行"
        return real(reports, keep_ids=keep_ids)

    monkeypatch.setattr(solver, "find_minimal_conflict", blocking)
    result: dict = {}

    def rejected_batch():
        with SessionLocal() as s:
            try:
                service.upload_batch(s, [r("X", 0, 2, 9)])
            except BatchConflictError as exc:
                result["conflict"] = exc.conflict_ids

    worker = threading.Thread(target=rejected_batch)
    worker.start()
    try:
        assert entered.wait(10), "冲突诊断未开始"
        # 诊断被卡住期间，另一笔自洽批次必须能完成入库。
        t0 = time.perf_counter()
        with SessionLocal() as s:
            out = service.upload_batch(s, [r("C", 50, 51, 3)])
        elapsed = time.perf_counter() - t0
        assert out["inserted_ids"] == ["C"]
        assert elapsed < 5
    finally:
        release.set()
        worker.join(10)
    assert result["conflict"] == ["A", "X"]
    assert _stored_totals() == {"A": 5, "C": 3}


def test_large_rejected_batch_with_interleaved_corrections(monkeypatch):
    """大批退回与更正交错：80 份现行 + 32 份新补录，冲突只涉及少数几份。"""
    gauges = [r(f"G{i:03d}", i * 30, i * 30 + 20, 1000) for i in range(80)]
    _seed(gauges)

    incoming = [r(f"N{i:02d}", 5000 + i, 5000 + i, 1) for i in range(30)]
    incoming += [
        r("BAD1", 30 * 77 + 1, 30 * 77 + 2, 5),
        r("BAD2", 30 * 77 + 1, 30 * 77 + 2, 7),
    ]

    captured: dict = {}
    real = solver.find_minimal_conflict

    def diagnose_with_correction(reports, keep_ids=None):
        captured["n"] = len(reports)
        # 交错：诊断过程中更正一份现行报表（同号整份替换，且保持自洽）。
        with SessionLocal() as s:
            service.upload_batch(s, [r("G000", 0, 20, 999)])
        return real(reports, keep_ids=keep_ids)

    monkeypatch.setattr(solver, "find_minimal_conflict", diagnose_with_correction)
    t0 = time.perf_counter()
    with SessionLocal() as s:
        with pytest.raises(BatchConflictError) as exc:
            service.upload_batch(s, incoming)
    elapsed = time.perf_counter() - t0

    assert captured["n"] == 112
    assert set(exc.value.conflict_ids) == {"BAD1", "BAD2"}
    stored = _stored_totals()
    assert stored["G000"] == 999  # 交错更正已入库
    assert not any(rid.startswith("N") or rid.startswith("BAD") for rid in stored)
    assert len(stored) == 80
    assert elapsed < 2, f"大批退回耗时过长: {elapsed:.2f}s"


def test_service_conflict_snapshot_is_independent_of_orm_rows(monkeypatch):
    """快照里的 Report 是不可变值对象，ORM 行后续变化不会影响诊断输入。"""
    _seed([r("A", 0, 2, 5)])
    seen: list = []
    real = solver.find_minimal_conflict

    def diagnose(reports, keep_ids=None):
        seen.append([(x.report_id, x.start_date, x.end_date, x.total) for x in reports])
        with SessionLocal() as s:
            row = s.get(ReportRow, "A")
            row.total = 77
            s.commit()
        return real(reports, keep_ids=keep_ids)

    monkeypatch.setattr(solver, "find_minimal_conflict", diagnose)
    with SessionLocal() as s:
        with pytest.raises(BatchConflictError):
            service.upload_batch(s, [r("X", 0, 2, 9)])
    assert ("A", d(0), d(2), 5) in seen[0]


def test_service_uses_no_solve_during_rejected_upload(monkeypatch):
    """退回路径全程只判可行性：连 solve 都不调用（查数缓存也不需要）。"""

    def _boom(*_args, **_kwargs):
        raise AssertionError("退回处理不应计算逐日人次")

    _seed([r("A", 0, 2, 5)])
    monkeypatch.setattr(solver, "solve", _boom)
    fresh = ReportService()
    with SessionLocal() as s:
        with pytest.raises(BatchConflictError):
            fresh.upload_batch(s, [r("X", 0, 2, 9)])
