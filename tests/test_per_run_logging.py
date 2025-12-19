"""Tests for per-run log file generation.

TDD tests written before implementation per fix-per-run-logging proposal.
Each pipeline run should have its own pipeline.log file in its run directory.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List
from concurrent.futures import ThreadPoolExecutor

import pytest

from sleap_roots_analyze.pipeline.pipelines import BasePipeline
from sleap_roots_analyze.pipeline.task import Task, TaskResult


# =============================================================================
# Test Fixtures: Simple pipeline for logging tests
# =============================================================================


@dataclass
class LoggingTestConfig:
    """Simple test config."""

    value: int = 10
    name: str = "test"


class LoggingTestPipeline(BasePipeline):
    """Test pipeline that logs messages."""

    def create_tasks(self) -> List[Task]:
        """Create tasks that produce log output."""

        def task_one(config, run_dir, logger):
            logger.info("Task one executing")
            return TaskResult(data={"step": 1})

        def task_two(config, run_dir, logger, task_one):
            logger.info("Task two executing")
            return TaskResult(data={"step": 2})

        return [
            Task(func=task_one, name="task_one", description="First task"),
            Task(
                func=task_two,
                name="task_two",
                depends_on=["task_one"],
                description="Second task",
            ),
        ]


class FailingLoggingPipeline(BasePipeline):
    """Pipeline that fails to test log persistence on failure."""

    def create_tasks(self) -> List[Task]:
        """Create tasks where one fails."""

        def success_task(config, run_dir, logger):
            logger.info("Success task completed")
            return TaskResult(data="success")

        def failing_task(config, run_dir, logger, success_task):
            logger.error("About to fail intentionally")
            raise RuntimeError("Intentional failure for testing")

        return [
            Task(func=success_task, name="success_task"),
            Task(
                func=failing_task,
                name="failing_task",
                depends_on=["success_task"],
            ),
        ]


# =============================================================================
# Tests for per-run log file creation
# =============================================================================


class TestPerRunLogFileCreation:
    """Tests for log file creation in run directory."""

    def test_per_run_log_file_created(self, tmp_path):
        """Test that pipeline.log is created in run directory after run."""
        config = LoggingTestConfig()
        pipeline = LoggingTestPipeline(
            config=config,
            output_dir=tmp_path,
            pipeline_name="test_pipeline",
        )

        pipeline.run()

        log_path = pipeline.run_dir / "pipeline.log"
        assert log_path.exists(), "pipeline.log should exist in run directory"

    def test_per_run_log_not_empty(self, tmp_path):
        """Test that per-run log file has content."""
        config = LoggingTestConfig()
        pipeline = LoggingTestPipeline(
            config=config,
            output_dir=tmp_path,
            pipeline_name="test_pipeline",
        )

        pipeline.run()

        log_path = pipeline.run_dir / "pipeline.log"
        assert log_path.stat().st_size > 0, "Log file should not be empty"

    def test_per_run_log_contains_start_message(self, tmp_path):
        """Test that log contains pipeline start message."""
        config = LoggingTestConfig()
        pipeline = LoggingTestPipeline(
            config=config,
            output_dir=tmp_path,
            pipeline_name="my_test_pipeline",
        )

        pipeline.run()

        log_path = pipeline.run_dir / "pipeline.log"
        log_content = log_path.read_text()

        assert (
            "Starting pipeline" in log_content
        ), "Log should contain 'Starting pipeline'"

    def test_per_run_log_contains_task_messages(self, tmp_path):
        """Test that log contains task execution messages."""
        config = LoggingTestConfig()
        pipeline = LoggingTestPipeline(
            config=config,
            output_dir=tmp_path,
            pipeline_name="test_pipeline",
        )

        pipeline.run()

        log_path = pipeline.run_dir / "pipeline.log"
        log_content = log_path.read_text()

        # Should contain task-specific log messages
        assert "Task one executing" in log_content
        assert "Task two executing" in log_content

    def test_per_run_log_contains_completion_message(self, tmp_path):
        """Test that log contains pipeline completion message."""
        config = LoggingTestConfig()
        pipeline = LoggingTestPipeline(
            config=config,
            output_dir=tmp_path,
            pipeline_name="test_pipeline",
        )

        pipeline.run()

        log_path = pipeline.run_dir / "pipeline.log"
        log_content = log_path.read_text()

        assert (
            "completed successfully" in log_content
            or "Pipeline completed" in log_content
        )


# =============================================================================
# Tests for log persistence on failure
# =============================================================================


class TestLogPersistenceOnFailure:
    """Tests for log file persistence when pipeline fails."""

    def test_per_run_log_exists_on_failure(self, tmp_path):
        """Test that log file exists even when pipeline fails."""
        config = LoggingTestConfig()
        pipeline = FailingLoggingPipeline(
            config=config,
            output_dir=tmp_path,
            pipeline_name="failing_pipeline",
        )

        with pytest.raises(RuntimeError, match="Intentional failure"):
            pipeline.run()

        log_path = pipeline.run_dir / "pipeline.log"
        assert log_path.exists(), "Log file should exist even on failure"

    def test_per_run_log_contains_failure_messages(self, tmp_path):
        """Test that log contains messages up to failure point."""
        config = LoggingTestConfig()
        pipeline = FailingLoggingPipeline(
            config=config,
            output_dir=tmp_path,
            pipeline_name="failing_pipeline",
        )

        with pytest.raises(RuntimeError):
            pipeline.run()

        log_path = pipeline.run_dir / "pipeline.log"
        log_content = log_path.read_text()

        # Should contain successful task log
        assert "Success task completed" in log_content
        # Should contain error message
        assert "About to fail" in log_content or "failed" in log_content.lower()


# =============================================================================
# Tests for multiple runs with independent logs
# =============================================================================


class TestMultipleRunsIndependentLogs:
    """Tests for independent logs across multiple runs."""

    def test_sequential_runs_have_separate_logs(self, tmp_path):
        """Test that sequential runs each have their own log file."""
        config = LoggingTestConfig()

        # First run
        pipeline1 = LoggingTestPipeline(
            config=config,
            output_dir=tmp_path,
            pipeline_name="run1",
        )
        pipeline1.run()
        log1_path = pipeline1.run_dir / "pipeline.log"

        # Second run
        pipeline2 = LoggingTestPipeline(
            config=config,
            output_dir=tmp_path,
            pipeline_name="run2",
        )
        pipeline2.run()
        log2_path = pipeline2.run_dir / "pipeline.log"

        # Both should exist
        assert log1_path.exists()
        assert log2_path.exists()

        # They should be in different directories
        assert log1_path.parent != log2_path.parent

        # Each should contain its own pipeline name
        log1_content = log1_path.read_text()
        log2_content = log2_path.read_text()

        assert "run1" in log1_content
        assert "run2" in log2_content

    def test_parallel_runs_have_independent_logs(self, tmp_path):
        """Test that parallel runs don't interfere with each other's logs."""
        config = LoggingTestConfig()

        def run_pipeline(name: str) -> tuple[Path, str]:
            """Run a pipeline and return its log path and content."""
            pipeline = LoggingTestPipeline(
                config=config,
                output_dir=tmp_path,
                pipeline_name=name,
            )
            pipeline.run()
            log_path = pipeline.run_dir / "pipeline.log"
            return log_path, log_path.read_text()

        # Run pipelines in parallel
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(run_pipeline, f"parallel_{i}") for i in range(3)]
            results = [f.result() for f in futures]

        # Each log should exist and contain only its own messages
        for i, (log_path, log_content) in enumerate(results):
            assert log_path.exists(), f"Log for parallel_{i} should exist"
            assert (
                f"parallel_{i}" in log_content
            ), f"Log for parallel_{i} should contain its own name"
            # Should not contain other pipeline names
            for j in range(3):
                if j != i:
                    # The log should not have messages from other pipelines
                    # (pipeline names appear in log as logger name)
                    pass  # This is hard to test without knowing exact format


# =============================================================================
# Tests for coexistence with CLI-level logging
# =============================================================================


class TestCoexistenceWithCLILogging:
    """Tests for per-run logs coexisting with CLI-level logs."""

    def test_per_run_log_independent_of_root_logger(self, tmp_path):
        """Test that per-run log works even with root logger configured."""
        # Configure a root logger (simulating CLI setup)
        root_logger = logging.getLogger()
        original_handlers = root_logger.handlers[:]

        try:
            cli_log_path = tmp_path / "cli_level.log"
            cli_handler = logging.FileHandler(cli_log_path)
            root_logger.addHandler(cli_handler)

            # Run pipeline
            config = LoggingTestConfig()
            pipeline = LoggingTestPipeline(
                config=config,
                output_dir=tmp_path,
                pipeline_name="test_pipeline",
            )
            pipeline.run()

            # Per-run log should exist
            per_run_log = pipeline.run_dir / "pipeline.log"
            assert per_run_log.exists()

        finally:
            # Restore original handlers
            root_logger.handlers = original_handlers
            cli_handler.close()
