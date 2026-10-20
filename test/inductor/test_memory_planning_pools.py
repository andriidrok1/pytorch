# Owner(s): ["module: inductor"]

import sys
from types import SimpleNamespace

import sympy

import torch
from torch._inductor import config, ir
from torch._inductor.codegen.cpp_utils import cexpr
from torch._inductor.codegen.memory_planning import (
    Allocation,
    AllocationPool,
    Empty,
    LiveRange,
    MemoryPlanner,
    SpatialSplit,
    TemporalSplit,
)
from torch._inductor.test_case import run_tests, TestCase


class MemoryPlanningPoolsTest(TestCase):
    class FakeBuffer:
        def __init__(self, name):
            self._name = name

        def get_device(self):
            return torch.device("cpu")

        def get_layout(self):
            return SimpleNamespace(size=())

        def get_name(self):
            return self._name

    class FakeBufferGroup:
        def __init__(self, allocation, is_output):
            self.allocation = allocation
            self.is_output = is_output

        def make_allocation(self):
            pass

    def make_group(self, name, size, is_output):
        allocation = Allocation(
            self.FakeBuffer(name),
            LiveRange(0, 10),
            size_hint=size,
            symbolic_size=size,
        )
        return self.FakeBufferGroup(allocation, is_output)

    def test_memory_pool_modes(self):
        expected_patterns = (
            ("none", (0, 1, 2, 3)),
            ("intermediates", (0, 1, 2, 2)),
            ("outputs", (0, 0, 1, 1)),
            ("combined", (0, 0, 0, 0)),
        )

        for memory_pool, expected_pool_pattern in expected_patterns:
            with self.subTest(memory_pool=memory_pool):
                groups = [
                    self.make_group("output_0", 16, True),
                    self.make_group("output_1", 32, True),
                    self.make_group("intermediate_0", 64, False),
                    self.make_group("intermediate_1", 48, False),
                ]
                planner = MemoryPlanner(wrapper=None)
                planner.buffer_groups = groups

                with config.patch(memory_pool=memory_pool):
                    planner.allocate_groups()

                pool_indices = {}
                actual_pool_pattern = tuple(
                    pool_indices.setdefault(
                        id(group.allocation.pool), len(pool_indices)
                    )
                    for group in groups
                )
                self.assertEqual(expected_pool_pattern, actual_pool_pattern)


class PoolAlignmentTest(TestCase):
    @staticmethod
    def make_allocation(name, nbytes, live_range):
        node = ir.Buffer(
            name=name,
            layout=ir.FixedLayout(torch.device("cpu"), torch.uint8, [nbytes], [1]),
        )
        return Allocation(node, live_range, size_hint=64, symbolic_size=nbytes)

    def make_append_pool(self, count, symbols=None):
        if symbols is None:
            symbols = sympy.symbols("s0 s1 s2", integer=True, nonnegative=True)
        x, y, z = symbols
        first = self.make_allocation("first", 2 * x, LiveRange(0, 1))
        second = self.make_allocation("second", 3 * y, LiveRange(1, 2))
        first.mark_allocated()
        second.mark_allocated()
        pool = AllocationPool(torch.device("cpu"), TemporalSplit([first, second]))
        blocks = [first, second]
        for index in range(count):
            block = self.make_allocation(f"buffer{index}", z, LiveRange(0, 2))
            pool.allocate_at_end(block)
            blocks.append(block)
        pool.finalize("pool0")
        return pool, blocks

    def make_strided_pool(self, count):
        size = sympy.Symbol("strided_size", integer=True, positive=True)
        blocks = []
        for index in range(count + 1):
            layout = ir.FixedLayout(
                torch.device("cpu"), torch.float32, [size, 7], [8, 1]
            )
            blocks.append(
                Allocation(
                    ir.Buffer(name=f"strided{index}", layout=layout),
                    LiveRange(0, 2),
                    size_hint=64,
                    symbolic_size=layout.storage_size() * torch.float32.itemsize,
                )
            )
        blocks[0].mark_allocated()
        pool = AllocationPool(torch.device("cpu"), TemporalSplit([blocks[0]]))
        for block in blocks[1:]:
            pool.allocate_at_end(block)
        pool.finalize("pool0")
        return pool, blocks

    def assert_pool_printable_at_2000(self, make_pool):
        previous = sys.getrecursionlimit()
        try:
            sys.setrecursionlimit(10000)
            pool, blocks = make_pool()
            nbytes = pool.root.get_symbolic_size()
            sys.setrecursionlimit(2000)
            self.assertEqual(sys.getrecursionlimit(), 2000)
            try:
                rendered = cexpr(nbytes, simplify=False)
                self.assertIsNotNone(blocks[-1].offset)
                offset_rendered = cexpr(blocks[-1].offset, simplify=False)
            except RecursionError:
                raise AssertionError(
                    "Pool-size code generation exceeded recursion limit 2000"
                ) from None
            self.assertGreater(len(rendered), 0)
            self.assertGreater(len(offset_rendered), 0)
        finally:
            sys.setrecursionlimit(previous)
        return pool, blocks, rendered

    def test_deep_pool_size_can_be_printed_at_2000(self):
        _, _, rendered = self.assert_pool_printable_at_2000(
            lambda: self.make_append_pool(220)
        )
        self.assertIn("std::max", rendered)
        self.assertIn("align", rendered)

    def test_strided_pool_size_can_be_printed_at_2000(self):
        pool, blocks, rendered = self.assert_pool_printable_at_2000(
            lambda: self.make_strided_pool(220)
        )
        self.assertIn("align", rendered)
        size = blocks[0].node.get_layout().size[0]
        self.assertEqual(blocks[0].get_symbolic_size(), 32 * size - 4)
        for value in (1, 2, 7, 33):
            with self.subTest(size=value):
                nbytes = 32 * value - 4
                step = 64 * ((nbytes + 63) // 64)
                self.assertEqual(
                    int(pool.root.get_symbolic_size().subs(size, value)),
                    220 * step + nbytes,
                )
                self.assertEqual(int(blocks[-1].offset.subs(size, value)), 220 * step)

    def test_pool_with_unknown_sign_sizes_can_be_printed_at_2000(self):
        symbols = sympy.symbols("x y z", integer=True)
        self.assertTrue(all(symbol.is_nonnegative is None for symbol in symbols))
        pool, blocks, rendered = self.assert_pool_printable_at_2000(
            lambda: self.make_append_pool(220, symbols)
        )
        self.assertIn("std::max", rendered)
        for values in ((1, 1, 1), (5, 7, 13)):
            with self.subTest(values=values):
                x, y, z = values
                substitutions = dict(zip(symbols, values))
                base = 64 * ((max(2 * x, 3 * y) + 63) // 64)
                step = 64 * ((z + 63) // 64)
                self.assertEqual(
                    int(pool.root.get_symbolic_size().subs(substitutions)),
                    base + 219 * step + z,
                )
                self.assertEqual(
                    int(blocks[-1].offset.subs(substitutions)), base + 219 * step
                )

    def test_pool_sizes_and_offsets_preserve_padding(self):
        x, y, z = sympy.symbols("s0 s1 s2", integer=True, nonnegative=True)
        pool, blocks = self.make_append_pool(3)
        for values in ((0, 0, 0), (1, 1, 1), (5, 7, 13), (32, 21, 64), (13, 32, 127)):
            with self.subTest(values=values):
                xv, yv, zv = values
                substitutions = dict(zip((x, y, z), values))
                base = 64 * ((max(2 * xv, 3 * yv) + 63) // 64)
                step = 64 * ((zv + 63) // 64)
                self.assertEqual(
                    int(pool.root.get_symbolic_size().subs(substitutions)),
                    base + 2 * step + zv,
                )
                self.assertEqual(
                    [
                        int(sympy.sympify(block.offset).subs(substitutions))
                        for block in blocks
                    ],
                    [0, 0, base, base + step, base + 2 * step],
                )

    def test_empty_pool_sizes(self):
        empty = TemporalSplit([])
        self.assertEqual(empty.get_symbolic_size(), 0)
        self.assertEqual(Empty(64).get_symbolic_size(), 0)
        spatial = SpatialSplit(empty, TemporalSplit([Empty(64)]))
        outer = SpatialSplit(TemporalSplit([spatial]), TemporalSplit([Empty(64)]))
        self.assertEqual(outer.get_symbolic_size(), 0)
        self.assertEqual(cexpr(outer.get_symbolic_size(), simplify=False), "0L")

    def test_single_allocation_pool_size_and_offset(self):
        size = sympy.Symbol("size", integer=True, nonnegative=True)
        block = self.make_allocation("single", size, LiveRange(0, 1))
        block.mark_allocated()
        pool = AllocationPool(torch.device("cpu"), TemporalSplit([block]))
        pool.finalize("pool0")
        self.assertEqual(pool.root.get_symbolic_size(), size)
        self.assertEqual(block.offset, 0)

    def test_temporal_reuse_invalidates_cached_pool_sizes(self):
        x, y, z = sympy.symbols("x y z", integer=True, nonnegative=True)
        first = self.make_allocation("first", 4 * x, LiveRange(0, 1))
        tail = self.make_allocation("tail", 8 * z, LiveRange(0, 3))
        anchor = self.make_allocation("anchor", sympy.Integer(1), LiveRange(0, 3))
        first.mark_allocated()
        pool = AllocationPool(torch.device("cpu"), TemporalSplit([first]))
        pool.allocate_at_end(tail)
        pool.allocate_at_end(anchor)
        before = pool.root.get_symbolic_size()
        replacement = self.make_allocation("replacement", 12 * y, LiveRange(1, 2))
        self.assertTrue(pool.allocate(replacement, is_last=False))
        after = pool.root.get_symbolic_size()
        pool.finalize("pool0")
        for values in ((1, 7, 1), (64, 1, 3), (0, 0, 0)):
            with self.subTest(values=values):
                xv, yv, zv = values
                substitutions = dict(zip((x, y, z), values))
                first_prefix = 64 * ((4 * xv + 63) // 64)
                updated_prefix = 64 * ((max(4 * xv, 12 * yv) + 63) // 64)
                tail_prefix = 64 * ((8 * zv + 63) // 64)
                self.assertEqual(
                    int(before.subs(substitutions)), first_prefix + tail_prefix + 1
                )
                self.assertEqual(
                    int(after.subs(substitutions)), updated_prefix + tail_prefix + 1
                )
                self.assertEqual(first.offset, 0)
                self.assertEqual(replacement.offset, 0)
                self.assertEqual(int(tail.offset.subs(substitutions)), updated_prefix)
                self.assertEqual(
                    int(anchor.offset.subs(substitutions)), updated_prefix + tail_prefix
                )


if __name__ == "__main__":
    run_tests()
