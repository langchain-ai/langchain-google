"""Get a specific task from Google Tasks."""

from typing import Optional, Type

from langchain_core.callbacks import CallbackManagerForToolRun
from pydantic import BaseModel, Field

from langchain_google_community.tasks.base import TasksBaseTool


class GetTaskSchema(BaseModel):
    """Input schema for `TasksGetTask`."""

    task_id: str = Field(..., description="The ID of the task to retrieve.")

    tasklist: str = Field(
        default="@default",
        description=(
            "The task list ID containing the task. "
            "Use '@default' for the default task list."
        ),
    )


class TasksGetTask(TasksBaseTool):  # type: ignore[override]
    """Tool that retrieves a specific task from Google Tasks."""

    name: str = "get_google_task"

    description: str = (
        "Use this tool to get detailed information about a specific task "
        "from Google Tasks. You must provide the task ID."
    )

    args_schema: Type[GetTaskSchema] = GetTaskSchema

    def _run(
        self,
        task_id: str,
        tasklist: str = "@default",
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Get a specific task from Google Tasks.

        Args:
            task_id: The ID of the task to retrieve.
            tasklist: The task list ID. Defaults to '@default'.
            run_manager: Optional callback manager.

        Returns:
            A formatted string with the task details.
        """
        try:
            task = (
                self.api_resource.tasks().get(tasklist=tasklist, task=task_id).execute()
            )

            response = "Task Details:\n"
            response += f"ID: {task.get('id', 'Unknown')}\n"
            response += f"Title: {task.get('title', 'No title')}\n"
            response += f"Status: {task.get('status', 'Unknown')}\n"

            if task.get("notes"):
                response += f"Notes: {task['notes']}\n"

            if task.get("due"):
                response += f"Due: {task['due']}\n"

            if task.get("updated"):
                response += f"Last Updated: {task['updated']}\n"

            if task.get("completed"):
                response += f"Completed: {task['completed']}\n"

            return response

        except Exception as e:
            return f"An error occurred while retrieving the task: {str(e)}"
