"""Delete a task from Google Tasks."""

from typing import Optional, Type

from langchain_core.callbacks import CallbackManagerForToolRun
from pydantic import BaseModel, Field

from langchain_google_community.tasks.base import TasksBaseTool


class DeleteTaskSchema(BaseModel):
    """Input schema for `TasksDeleteTask`."""

    task_id: str = Field(..., description="The ID of the task to delete.")

    tasklist: str = Field(
        default="@default",
        description=(
            "The task list ID containing the task. "
            "Use '@default' for the default task list."
        ),
    )


class TasksDeleteTask(TasksBaseTool):  # type: ignore[override]
    """Tool that deletes a task from Google Tasks."""

    name: str = "delete_google_task"

    description: str = (
        "Use this tool to delete a task from Google Tasks. "
        "You must provide the task ID."
    )

    args_schema: Type[DeleteTaskSchema] = DeleteTaskSchema

    def _run(
        self,
        task_id: str,
        tasklist: str = "@default",
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Delete a task from Google Tasks.

        Args:
            task_id: The ID of the task to delete.
            tasklist: The task list ID. Defaults to '@default'.
            run_manager: Optional callback manager.

        Returns:
            A string confirming the task deletion.
        """
        try:
            self.api_resource.tasks().delete(tasklist=tasklist, task=task_id).execute()

            return f"Task with ID '{task_id}' has been deleted successfully."

        except Exception as e:
            return f"An error occurred while deleting the task: {str(e)}"
