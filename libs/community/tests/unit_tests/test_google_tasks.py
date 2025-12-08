"""Unit tests for Google Tasks tools."""

from unittest.mock import MagicMock

import pytest

from langchain_google_community.tasks.create_task import TasksCreateTask
from langchain_google_community.tasks.delete_task import TasksDeleteTask
from langchain_google_community.tasks.get_task import TasksGetTask
from langchain_google_community.tasks.list_tasks import TasksListTasks
from langchain_google_community.tasks.update_task import TasksUpdateTask


def test_create_task() -> None:
    """Test creating a task."""
    mock_api_resource = MagicMock()
    mock_api_resource.tasks().insert().execute.return_value = {
        "id": "task123",
        "title": "Test Task",
        "notes": "Test notes",
    }

    # Bypass pydantic validation as google-api-python-client is not a dependency
    tool = TasksCreateTask.model_construct(api_resource=mock_api_resource)
    tool_input = {
        "title": "Test Task",
        "notes": "Test notes",
    }
    result = tool.run(tool_input)
    assert tool.args_schema is not None
    assert "Task created successfully!" in result
    assert "task123" in result
    assert "Test Task" in result


def test_create_task_with_due_date() -> None:
    """Test creating a task with a due date."""
    mock_api_resource = MagicMock()
    mock_api_resource.tasks().insert().execute.return_value = {
        "id": "task456",
        "title": "Task with due date",
        "due": "2025-12-31T23:59:59Z",
    }

    tool = TasksCreateTask.model_construct(api_resource=mock_api_resource)
    tool_input = {
        "title": "Task with due date",
        "due": "2025-12-31T23:59:59Z",
    }
    result = tool.run(tool_input)
    assert "Task created successfully!" in result
    assert "2025-12-31T23:59:59Z" in result


def test_list_tasks() -> None:
    """Test listing tasks."""
    mock_api_resource = MagicMock()
    mock_api_resource.tasks().list().execute.return_value = {
        "items": [
            {
                "id": "task1",
                "title": "First Task",
                "status": "needsAction",
            },
            {
                "id": "task2",
                "title": "Second Task",
                "status": "completed",
                "notes": "Some notes here",
            },
        ]
    }

    tool = TasksListTasks.model_construct(api_resource=mock_api_resource)
    tool_input = {"max_results": 10}
    result = tool.run(tool_input)
    assert tool.args_schema is not None
    assert "Found 2 task(s)" in result
    assert "First Task" in result
    assert "Second Task" in result


def test_list_tasks_empty() -> None:
    """Test listing tasks when no tasks exist."""
    mock_api_resource = MagicMock()
    mock_api_resource.tasks().list().execute.return_value = {"items": []}

    tool = TasksListTasks.model_construct(api_resource=mock_api_resource)
    tool_input: dict = {}
    result = tool.run(tool_input)
    assert "No tasks found" in result


def test_update_task() -> None:
    """Test updating a task."""
    mock_api_resource = MagicMock()
    mock_api_resource.tasks().get().execute.return_value = {
        "id": "task123",
        "title": "Old Title",
        "status": "needsAction",
    }
    mock_api_resource.tasks().update().execute.return_value = {
        "id": "task123",
        "title": "New Title",
        "status": "completed",
    }

    tool = TasksUpdateTask.model_construct(api_resource=mock_api_resource)
    tool_input = {
        "task_id": "task123",
        "title": "New Title",
        "status": "completed",
    }
    result = tool.run(tool_input)
    assert tool.args_schema is not None
    assert "Task updated successfully!" in result
    assert "New Title" in result
    assert "completed" in result


def test_delete_task() -> None:
    """Test deleting a task."""
    mock_api_resource = MagicMock()
    mock_api_resource.tasks().delete().execute.return_value = None

    tool = TasksDeleteTask.model_construct(api_resource=mock_api_resource)
    tool_input = {"task_id": "task123"}
    result = tool.run(tool_input)
    assert tool.args_schema is not None
    assert "deleted successfully" in result
    assert "task123" in result


def test_get_task() -> None:
    """Test getting a specific task."""
    mock_api_resource = MagicMock()
    mock_api_resource.tasks().get().execute.return_value = {
        "id": "task123",
        "title": "Sample Task",
        "status": "needsAction",
        "notes": "Task notes",
        "due": "2025-12-31T23:59:59Z",
    }

    tool = TasksGetTask.model_construct(api_resource=mock_api_resource)
    tool_input = {"task_id": "task123"}
    result = tool.run(tool_input)
    assert tool.args_schema is not None
    assert "Task Details:" in result
    assert "Sample Task" in result
    assert "needsAction" in result
    assert "Task notes" in result


def test_toolkit() -> None:
    """Test the TasksToolkit."""
    mock_api_resource = MagicMock()
    # Create individual tools with model_construct
    create_tool = TasksCreateTask.model_construct(api_resource=mock_api_resource)
    list_tool = TasksListTasks.model_construct(api_resource=mock_api_resource)
    update_tool = TasksUpdateTask.model_construct(api_resource=mock_api_resource)
    delete_tool = TasksDeleteTask.model_construct(api_resource=mock_api_resource)
    get_tool = TasksGetTask.model_construct(api_resource=mock_api_resource)

    # Verify tools can be created
    assert create_tool.name == "create_google_task"
    assert list_tool.name == "list_google_tasks"
    assert update_tool.name == "update_google_task"
    assert delete_tool.name == "delete_google_task"
    assert get_tool.name == "get_google_task"


@pytest.mark.requires("google-api-python-client")
def test_imports() -> None:
    """Test that all imports work correctly."""
    from langchain_google_community.tasks import (
        CreateTaskSchema,
        DeleteTaskSchema,
        GetTaskSchema,
        ListTasksSchema,
        TasksCreateTask,
        TasksDeleteTask,
        TasksGetTask,
        TasksListTasks,
        TasksToolkit,
        TasksUpdateTask,
        UpdateTaskSchema,
    )

    assert CreateTaskSchema is not None
    assert DeleteTaskSchema is not None
    assert GetTaskSchema is not None
    assert ListTasksSchema is not None
    assert TasksCreateTask is not None
    assert TasksDeleteTask is not None
    assert TasksGetTask is not None
    assert TasksListTasks is not None
    assert TasksToolkit is not None
    assert TasksUpdateTask is not None
    assert UpdateTaskSchema is not None
