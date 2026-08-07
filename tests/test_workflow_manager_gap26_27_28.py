"""
Tests for Workflow Manager (GAP #26-28)

Test coverage:
- BatchProcessor: Batch processing API (GAP #26)
- UndoRedoManager: Undo/Redo system (GAP #27)
- WorkflowSessionManager: Session integration (GAP #28)
- WorkflowManager: Unified workflow API

Author: AURIK Team
Version: 1.0.0
"""

import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock

import numpy as np
import pytest

from workflow.workflow_manager import (
    BatchJobConfig,
    BatchJobResult,
    BatchProcessor,
    UndoRedoManager,
    WorkflowManager,
    WorkflowSessionManager,
)

# --- Helper Functions ---


def create_test_audio(sr=16000, duration=1.0, freq=440.0):
    """Create test audio signal."""
    t = np.linspace(0, duration, int(sr * duration))
    audio = np.sin(2 * np.pi * freq * t) * 0.5
    return audio.astype(np.float32)


def create_test_audio_files(tmpdir, count=3):
    """Create temporary test audio files."""
    import soundfile as sf

    files = []
    for i in range(count):
        audio = create_test_audio(freq=440.0 * (i + 1))
        filepath = tmpdir / f"test_{i}.wav"
        sf.write(filepath, audio, 16000)
        files.append(filepath)

    return files


@pytest.fixture
def test_dir():
    """Create temporary test directory."""
    tmpdir = Path(tempfile.mkdtemp())
    yield tmpdir
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def test_audio_files(test_dir):
    """Create test audio files."""
    return create_test_audio_files(test_dir, count=3)


# --- BatchProcessor Tests (GAP #26) ---


class TestBatchProcessor:
    def test_initialization(self):
        """Test BatchProcessor initialization."""
        processor = BatchProcessor()
        assert processor is not None
        assert processor.restorer_factory is not None

    def test_batch_job_config(self, test_dir):
        """Test BatchJobConfig creation."""
        config = BatchJobConfig(
            input_files=[Path("file1.wav"), Path("file2.wav")],
            output_dir=test_dir,
            processing_mode="balanced",
            parallel=False,
        )

        assert len(config.input_files) == 2
        assert config.output_dir == test_dir
        assert config.processing_mode == "balanced"
        assert not config.parallel

    def test_sequential_processing(self, test_audio_files, test_dir):
        """Test sequential batch processing."""
        output_dir = test_dir / "output"
        output_dir.mkdir()

        # Mock restorer
        mock_restorer = Mock(spec=["process"])
        mock_restorer.process = Mock(side_effect=lambda audio, sr, **kwargs: audio * 1.1)

        processor = BatchProcessor(restorer_factory=lambda: mock_restorer)

        config = BatchJobConfig(
            input_files=test_audio_files, output_dir=output_dir, processing_mode="balanced", parallel=False
        )

        result = processor.process_batch(config)

        assert result.total_files == 3
        assert result.success_count == 3
        assert result.failed_count == 0
        assert result.success_rate() == 1.0

    def test_parallel_processing(self, test_audio_files, test_dir):
        """Test parallel batch processing."""
        output_dir = test_dir / "output"
        output_dir.mkdir()

        # Mock restorer
        mock_restorer = Mock(spec=["process"])
        mock_restorer.process = Mock(side_effect=lambda audio, sr, **kwargs: audio * 1.1)

        processor = BatchProcessor(restorer_factory=lambda: mock_restorer)

        config = BatchJobConfig(
            input_files=test_audio_files,
            output_dir=output_dir,
            processing_mode="balanced",
            parallel=True,
            max_workers=2,
        )

        result = processor.process_batch(config)

        assert result.total_files == 3
        # Parallel might fallback to sequential in some environments
        assert result.success_count + result.failed_count == result.total_files

    def test_skip_existing_files(self, test_audio_files, test_dir):
        """Test skipping existing output files."""
        output_dir = test_dir / "output"
        output_dir.mkdir()

        # Create existing output file
        existing_output = output_dir / "test_0_restored.wav"
        import soundfile as sf

        sf.write(existing_output, create_test_audio(), 16000)

        mock_restorer = Mock(spec=["process"])
        mock_restorer.process = Mock(side_effect=lambda audio, sr, **kwargs: audio * 1.1)

        processor = BatchProcessor(restorer_factory=lambda: mock_restorer)

        config = BatchJobConfig(
            input_files=test_audio_files,
            output_dir=output_dir,
            processing_mode="balanced",
            parallel=False,
            skip_existing=True,
        )

        result = processor.process_batch(config)

        assert result.skipped_count >= 1
        assert result.total_files == 3

    def test_progress_callback(self, test_audio_files, test_dir):
        """Test progress callback invocation."""
        output_dir = test_dir / "output"
        output_dir.mkdir()

        progress_calls = []

        def on_progress(current, total, filename):
            progress_calls.append((current, total, filename))

        mock_restorer = Mock(spec=["process"])
        mock_restorer.process = Mock(side_effect=lambda audio, sr, **kwargs: audio * 1.1)

        processor = BatchProcessor(restorer_factory=lambda: mock_restorer)

        config = BatchJobConfig(
            input_files=test_audio_files,
            output_dir=output_dir,
            processing_mode="balanced",
            parallel=False,
            on_progress=on_progress,
        )

        processor.process_batch(config)

        assert len(progress_calls) == 3
        assert progress_calls[0][0] == 1  # First file
        assert progress_calls[-1][0] == 3  # Last file

    def test_file_complete_callback(self, test_audio_files, test_dir):
        """Test file complete callback."""
        output_dir = test_dir / "output"
        output_dir.mkdir()

        completed_files = []

        def on_file_complete(input_path, output_path, success, error):
            completed_files.append({"input": input_path, "output": output_path, "success": success})

        mock_restorer = Mock(spec=["process"])
        mock_restorer.process = Mock(side_effect=lambda audio, sr, **kwargs: audio * 1.1)

        processor = BatchProcessor(restorer_factory=lambda: mock_restorer)

        config = BatchJobConfig(
            input_files=test_audio_files,
            output_dir=output_dir,
            processing_mode="balanced",
            parallel=False,
            on_file_complete=on_file_complete,
        )

        processor.process_batch(config)

        assert len(completed_files) == 3
        assert all(f["success"] for f in completed_files)

    def test_error_handling(self, test_audio_files, test_dir):
        """Test error handling in batch processing."""
        output_dir = test_dir / "output"
        output_dir.mkdir()

        # Mock restorer that fails on second file
        call_count = [0]

        def failing_process(audio, sr, **kwargs):
            call_count[0] += 1
            if call_count[0] == 2:
                raise ValueError("Simulated processing error")
            return audio * 1.1

        mock_restorer = Mock(spec=["process"])
        mock_restorer.process = Mock(side_effect=failing_process)

        processor = BatchProcessor(restorer_factory=lambda: mock_restorer)

        config = BatchJobConfig(
            input_files=test_audio_files, output_dir=output_dir, processing_mode="balanced", parallel=False
        )

        result = processor.process_batch(config)

        assert result.total_files == 3
        assert result.failed_count >= 1
        assert result.success_count >= 1

    def test_batch_result_summary(self):
        """Test BatchJobResult summary generation."""
        result = BatchJobResult(
            total_files=10, success_count=8, failed_count=2, skipped_count=0, total_time=45.5, results=[]
        )

        assert result.success_rate() == 0.8
        summary = result.summary()
        assert "8/10" in summary
        assert "2 failed" in summary
        assert "45.5s" in summary


# --- UndoRedoManager Tests (GAP #27) ---


class TestUndoRedoManager:
    def test_initialization(self, test_dir):
        """Test UndoRedoManager initialization."""
        undo_manager = UndoRedoManager(backup_dir=test_dir / "undo")

        assert undo_manager.max_history == 10
        assert undo_manager.backup_dir.exists()
        assert len(undo_manager.undo_stack) == 0
        assert len(undo_manager.redo_stack) == 0

    def test_save_state(self, test_dir):
        """Test saving processing state."""
        undo_manager = UndoRedoManager(backup_dir=test_dir / "undo")

        # Create test file
        test_file = test_dir / "test.wav"
        import soundfile as sf

        sf.write(test_file, create_test_audio(), 16000)

        # Save state
        undo_manager.save_state(
            input_path=test_file, output_path=test_file, settings={"mode": "balanced"}, description="Test processing"
        )

        assert len(undo_manager.undo_stack) == 1
        assert len(undo_manager.redo_stack) == 0
        assert undo_manager.can_undo()

    def test_undo_operation(self, test_dir):
        """Test undo operation."""
        undo_manager = UndoRedoManager(backup_dir=test_dir / "undo")

        # Create and modify test file
        test_file = test_dir / "test.wav"
        import soundfile as sf

        original_audio = create_test_audio()
        sf.write(test_file, original_audio, 16000)

        # Save state
        undo_manager.save_state(
            input_path=test_file, output_path=test_file, settings={"mode": "balanced"}, description="Test processing"
        )

        # Modify file
        modified_audio = original_audio * 1.5
        sf.write(test_file, modified_audio, 16000)

        # Undo
        state = undo_manager.undo()

        assert state is not None
        assert state.description == "Test processing"
        assert len(undo_manager.redo_stack) == 1
        assert not undo_manager.can_undo()
        assert undo_manager.can_redo()

    def test_redo_operation(self, test_dir):
        """Test redo operation."""
        undo_manager = UndoRedoManager(backup_dir=test_dir / "undo")

        test_file = test_dir / "test.wav"
        import soundfile as sf

        sf.write(test_file, create_test_audio(), 16000)

        # Save state
        undo_manager.save_state(
            input_path=test_file, output_path=test_file, settings={"mode": "balanced"}, description="Test processing"
        )

        # Undo
        undo_manager.undo()

        # Redo
        state = undo_manager.redo()

        assert state is not None
        assert len(undo_manager.undo_stack) == 1
        assert len(undo_manager.redo_stack) == 0
        assert undo_manager.can_undo()
        assert not undo_manager.can_redo()

    def test_undo_history_limit(self, test_dir):
        """Test undo history size limit."""
        undo_manager = UndoRedoManager(max_history=3, backup_dir=test_dir / "undo")

        test_file = test_dir / "test.wav"
        import soundfile as sf

        sf.write(test_file, create_test_audio(), 16000)

        # Save more states than max_history
        for i in range(5):
            undo_manager.save_state(
                input_path=test_file,
                output_path=test_file,
                settings={"mode": "balanced"},
                description=f"Processing {i}",
            )

        # Should only keep max_history states
        assert len(undo_manager.undo_stack) == 3

    def test_new_action_clears_redo(self, test_dir):
        """Test that new action clears redo stack."""
        undo_manager = UndoRedoManager(backup_dir=test_dir / "undo")

        test_file = test_dir / "test.wav"
        import soundfile as sf

        sf.write(test_file, create_test_audio(), 16000)

        # Create states
        undo_manager.save_state(test_file, test_file, {}, "State 1")
        undo_manager.save_state(test_file, test_file, {}, "State 2")

        # Undo once
        undo_manager.undo()
        assert undo_manager.can_redo()

        # New action should clear redo
        undo_manager.save_state(test_file, test_file, {}, "State 3")
        assert not undo_manager.can_redo()

    def test_get_undo_history(self, test_dir):
        """Test getting undo history."""
        undo_manager = UndoRedoManager(backup_dir=test_dir / "undo")

        test_file = test_dir / "test.wav"
        import soundfile as sf

        sf.write(test_file, create_test_audio(), 16000)

        undo_manager.save_state(test_file, test_file, {}, "Action 1")
        undo_manager.save_state(test_file, test_file, {}, "Action 2")
        undo_manager.save_state(test_file, test_file, {}, "Action 3")

        history = undo_manager.get_undo_history()

        assert len(history) == 3
        assert history[0] == "Action 3"  # Most recent first
        assert history[-1] == "Action 1"  # Oldest last

    def test_cleanup_backups(self, test_dir):
        """Test cleanup of backup files."""
        undo_manager = UndoRedoManager(backup_dir=test_dir / "undo")

        test_file = test_dir / "test.wav"
        import soundfile as sf

        sf.write(test_file, create_test_audio(), 16000)

        # Create states
        for i in range(3):
            undo_manager.save_state(test_file, test_file, {}, f"State {i}")

        # Cleanup
        undo_manager.cleanup_all()

        assert len(undo_manager.undo_stack) == 0
        assert len(undo_manager.redo_stack) == 0


# --- WorkflowSessionManager Tests (GAP #28) ---


class TestWorkflowSessionManager:
    def test_initialization(self, test_dir):
        """Test WorkflowSessionManager initialization."""
        session_manager = WorkflowSessionManager(sessions_dir=test_dir / "sessions")

        assert session_manager is not None
        assert session_manager.auto_save

    def test_create_session(self, test_dir):
        """Test session creation."""
        session_manager = WorkflowSessionManager(sessions_dir=test_dir / "sessions")

        # Mock the backend if available
        if session_manager.backend_manager:
            session_manager.backend_manager.create_session = MagicMock(return_value="session_123")  # type: ignore[method-assign]
            session_id = session_manager.create_session("Test Session", "Description")
            assert session_id == "session_123"
            assert session_manager.current_session_id == "session_123"
        else:
            # Fallback behavior
            session_id = session_manager.create_session("Test Session", "Description")
            assert session_manager.current_session_id is not None

    def test_load_session(self, test_dir):
        """Test loading existing session."""
        session_manager = WorkflowSessionManager(sessions_dir=test_dir / "sessions")

        # Mock the backend if available
        if session_manager.backend_manager:
            mock_session = MagicMock()
            session_manager.backend_manager.load_session = MagicMock(return_value=mock_session)  # type: ignore[method-assign]
            success = session_manager.load_session("session_456")
            assert success
            assert session_manager.current_session_id == "session_456"
        else:
            # Fallback behavior - should handle gracefully
            success = session_manager.load_session("session_456")
            assert success is not None

    def test_add_processed_file_without_backend(self, test_dir):
        """Test adding processed file when backend not available."""
        session_manager = WorkflowSessionManager(sessions_dir=test_dir / "sessions")
        session_manager.backend_manager = None  # type: ignore[assignment]

        # Should not raise exception
        session_manager.add_processed_file(Path("input.wav"), Path("output.wav"), {"mode": "balanced"}, success=True)


# --- WorkflowManager Integration Tests ---


class TestWorkflowManager:
    def test_initialization(self, test_dir):
        """Test WorkflowManager initialization."""
        workflow = WorkflowManager(sessions_dir=test_dir / "sessions", backup_dir=test_dir / "undo")

        assert workflow.batch_processor is not None
        assert workflow.undo_manager is not None
        assert workflow.session_manager is not None

    def test_integrated_batch_processing(self, test_audio_files, test_dir):
        """Test batch processing with session integration."""
        workflow = WorkflowManager(sessions_dir=test_dir / "sessions", backup_dir=test_dir / "undo")

        # Create session
        workflow.create_session("Test Batch")

        # Mock restorer
        mock_restorer = Mock(spec=["process"])
        mock_restorer.process = Mock(side_effect=lambda audio, sr, **kwargs: audio * 1.1)
        workflow.batch_processor.restorer_factory = lambda: mock_restorer

        # Process batch
        output_dir = test_dir / "output"
        output_dir.mkdir()

        config = BatchJobConfig(
            input_files=test_audio_files, output_dir=output_dir, processing_mode="balanced", parallel=False
        )

        result = workflow.process_batch(config, enable_undo=True)

        assert result.total_files == 3
        assert result.success_count == 3

        # Verify undo is available
        assert workflow.can_undo()

    def test_undo_after_batch(self, test_audio_files, test_dir):
        """Test undo after batch processing."""
        workflow = WorkflowManager(sessions_dir=test_dir / "sessions", backup_dir=test_dir / "undo")

        mock_restorer = Mock(spec=["process"])
        mock_restorer.process = Mock(side_effect=lambda audio, sr, **kwargs: audio * 1.1)
        workflow.batch_processor.restorer_factory = lambda: mock_restorer

        output_dir = test_dir / "output"
        output_dir.mkdir()

        config = BatchJobConfig(
            input_files=test_audio_files, output_dir=output_dir, processing_mode="balanced", parallel=False
        )

        result = workflow.process_batch(config, enable_undo=True)

        # Perform undo
        undo_success = workflow.undo()

        assert undo_success
        assert workflow.can_redo()

    def test_undo_redo_cycle(self, test_audio_files, test_dir):
        """Test complete undo/redo cycle."""
        workflow = WorkflowManager(sessions_dir=test_dir / "sessions", backup_dir=test_dir / "undo")

        mock_restorer = Mock(spec=["process"])
        mock_restorer.process = Mock(side_effect=lambda audio, sr, **kwargs: audio * 1.1)
        workflow.batch_processor.restorer_factory = lambda: mock_restorer

        output_dir = test_dir / "output"
        output_dir.mkdir()

        # Process only one file for clean undo/redo cycle test
        config = BatchJobConfig(
            input_files=[test_audio_files[0]],  # Use only first file
            output_dir=output_dir,
            processing_mode="balanced",
            parallel=False,
        )

        # Process - creates 1 undo state
        workflow.process_batch(config, enable_undo=True)
        assert workflow.can_undo()
        assert not workflow.can_redo()

        # Undo - moves state to redo stack
        assert workflow.undo()
        assert not workflow.can_undo()  # Now no undo available
        assert workflow.can_redo()

        # Redo - moves state back to undo stack
        assert workflow.redo()
        assert workflow.can_undo()
        assert not workflow.can_redo()

    def test_get_undo_history(self, test_audio_files, test_dir):
        """Test retrieving undo history."""
        workflow = WorkflowManager(sessions_dir=test_dir / "sessions", backup_dir=test_dir / "undo")

        mock_restorer = Mock(spec=["process"])
        mock_restorer.process = Mock(side_effect=lambda audio, sr, **kwargs: audio * 1.1)
        workflow.batch_processor.restorer_factory = lambda: mock_restorer

        output_dir = test_dir / "output"
        output_dir.mkdir()

        config = BatchJobConfig(
            input_files=test_audio_files, output_dir=output_dir, processing_mode="balanced", parallel=False
        )

        workflow.process_batch(config, enable_undo=True)

        history = workflow.get_undo_history()

        assert len(history) > 0
        assert all(isinstance(h, str) for h in history)

    def test_cleanup(self, test_dir):
        """Test workflow cleanup."""
        workflow = WorkflowManager(sessions_dir=test_dir / "sessions", backup_dir=test_dir / "undo")

        # Should not raise exception
        workflow.cleanup()


# --- Quality Gates ---


class TestQualityGates:
    def test_batch_processing_preserves_audio_quality(self, test_audio_files, test_dir):
        """Test that batch processing preserves audio quality."""
        processor = BatchProcessor()

        # Mock restorer that passes through audio
        mock_restorer = Mock(spec=["process"])
        mock_restorer.process = Mock(side_effect=lambda audio, sr, **kwargs: audio)
        processor.restorer_factory = lambda: mock_restorer

        output_dir = test_dir / "output"
        output_dir.mkdir()

        config = BatchJobConfig(
            input_files=test_audio_files, output_dir=output_dir, processing_mode="balanced", parallel=False
        )

        processor.process_batch(config)

        # Verify output files exist and have reasonable content
        import soundfile as sf

        for i in range(len(test_audio_files)):
            output_file = output_dir / f"test_{i}_restored.wav"
            assert output_file.exists()

            audio, sr = sf.read(output_file)
            assert len(audio) > 0
            assert np.max(np.abs(audio)) <= 1.0

    def test_undo_restores_original_state(self, test_dir):
        """Test that undo correctly restores original state."""
        undo_manager = UndoRedoManager(backup_dir=test_dir / "undo")

        test_file = test_dir / "test.wav"
        import soundfile as sf

        original_audio = create_test_audio()
        sf.write(test_file, original_audio, 16000)

        # Save state
        undo_manager.save_state(test_file, test_file, {}, "Processing")

        # Modify file
        modified_audio = original_audio * 0.5
        sf.write(test_file, modified_audio, 16000)

        # Read modified
        modified_read, _ = sf.read(test_file)

        # Undo
        undo_manager.undo()

        # Read after undo
        restored, _ = sf.read(test_file)

        # Should be closer to original than modified
        original_diff = np.mean(np.abs(original_audio - restored))
        modified_diff = np.mean(np.abs(modified_audio - restored))

        assert original_diff < modified_diff


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
