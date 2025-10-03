# Copyright © 2024-2025 Frello Technology Private Limited
# Copyright © 2025- Yash Bonde github.com/yashbonde
# MIT License

import asyncio
import pytest
from unittest import TestCase, main as ut_main

import tuneapi.types as tt


# Test tools for sync/async functionality
@tt.tool()
def sync_add(a: int, b: int) -> int:
    """Add two numbers synchronously."""
    return a + b


@tt.tool()
def sync_multiply(x: float, y: float) -> float:
    """Multiply two numbers synchronously."""
    return x * y


@tt.tool()
async def async_add(a: int, b: int) -> int:
    """Add two numbers asynchronously."""
    await asyncio.sleep(0.01)  # Simulate async work
    return a + b


@tt.tool()
async def async_multiply(x: float, y: float) -> float:
    """Multiply two numbers asynchronously."""
    await asyncio.sleep(0.01)  # Simulate async work
    return x * y


@tt.tool()
def sync_error_tool(should_error: bool) -> str:
    """Tool that can raise an error for testing."""
    if should_error:
        raise ValueError("Test error from sync tool")
    return "success"


@tt.tool()
async def async_error_tool(should_error: bool) -> str:
    """Async tool that can raise an error for testing."""
    await asyncio.sleep(0.01)
    if should_error:
        raise ValueError("Test error from async tool")
    return "success"


class TestToolCallSyncAsync(TestCase):
    """Test sync/async functionality of ToolCall methods."""

    def setUp(self):
        """Set up test fixtures."""
        self.sync_tool_call = tt.ToolCall(tool=sync_add, arguments={"a": 5, "b": 3})

        self.async_tool_call = tt.ToolCall(tool=async_add, arguments={"a": 10, "b": 7})

        self.sync_multiply_call = tt.ToolCall(
            tool=sync_multiply, arguments={"x": 2.5, "y": 4.0}
        )

        self.async_multiply_call = tt.ToolCall(
            tool=async_multiply, arguments={"x": 3.0, "y": 2.0}
        )

    def test_sync_tool_run(self):
        """Test that sync tools work with run() method."""
        result = self.sync_tool_call.run()
        self.assertEqual(result, 8)

        result = self.sync_multiply_call.run()
        self.assertEqual(result, 10.0)

    def test_sync_tool_run_async(self):
        """Test that sync tools work with run_async() method."""

        async def test_async():
            result = await self.sync_tool_call.run_async()
            self.assertEqual(result, 8)

            result = await self.sync_multiply_call.run_async()
            self.assertEqual(result, 10.0)

        asyncio.run(test_async())

    def test_async_tool_run_async(self):
        """Test that async tools work with run_async() method."""

        async def test_async():
            result = await self.async_tool_call.run_async()
            self.assertEqual(result, 17)

            result = await self.async_multiply_call.run_async()
            self.assertEqual(result, 6.0)

        asyncio.run(test_async())

    def test_mixed_sync_async_workflow(self):
        """Test a workflow mixing sync and async tools."""

        async def mixed_workflow():
            # Start with sync tool
            sync_result = await self.sync_tool_call.run_async()
            self.assertEqual(sync_result, 8)

            # Use result in async tool
            async_call = tt.ToolCall(
                tool=async_add, arguments={"a": sync_result, "b": 2}
            )
            async_result = await async_call.run_async()
            self.assertEqual(async_result, 10)

            # Back to sync tool
            final_call = tt.ToolCall(
                tool=sync_multiply, arguments={"x": float(async_result), "y": 1.5}
            )
            final_result = await final_call.run_async()
            self.assertEqual(final_result, 15.0)

        asyncio.run(mixed_workflow())
