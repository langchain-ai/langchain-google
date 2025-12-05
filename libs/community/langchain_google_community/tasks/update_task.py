"""Update a task in Google Tasks."""

from typing import Optional, Type

from langchain_core.callbacks import CallbackManagerForToolRun
from pydantic import BaseModel, Field

from langchain_google_community.tasks.base import TasksBaseTool


class UpdateTaskSchema(BaseModel):
    """Input schema for `TasksUpdateTask`."""

    task_id: str = Field(..., description="The ID of the task to update.")

    title: Optional[str] = Field(default=None, description="The new title of the task.")

    notes: Optional[str] = Field(
        default=None, description="The new notes or description for the task."
    )

    status: Optional[str] = Field(
        default=None,
        description=(
            "The new status of the task. Use 'completed' to mark as complete, "
            "or 'needsAction' to mark as incomplete."
        ),
    )

    due: Optional[str] = Field(
        default=None,
        description=(
            "The new due date in RFC 3339 format (e.g., '2024-12-31T23:59:59Z')."
        ),
    )

    tasklist: str = Field(
        default="@default",
        description=(
            "The task list ID containing the task. "
            "Use '@default' for the default task list."
        ),
    )


class TasksUpdateTask(TasksBaseTool):  # type: ignore[override]
    """Tool that updates a task in Google Tasks."""

    name: str = "update_google_task"

    description: str = (
        "Use this tool to update an existing task in Google Tasks. "
        "You can update the title, notes, status (completed/needsAction), or due date. "
        "You must provide the task ID."
    )

    args_schema: Type[UpdateTaskSchema] = UpdateTaskSchema

    def _run(
        self,
        task_id: str,
        title: Optional[str] = None,
        notes: Optional[str] = None,
        status: Optional[str] = None,
        due: Optional[str] = None,
        tasklist: str = "@default",
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Update a task in Google Tasks.

        Args:
            task_id: The ID of the task to update.
            title: Optional new title.
            notes: Optional new notes.
            status: Optional new status ('completed' or 'needsAction').
            due: Optional new due date in RFC 3339 format.
            tasklist: The task list ID. Defaults to '@default'.
            run_manager: Optional callback manager.

        Returns:
            A string confirming the task update with updated details.
        """
        try:
            # First, get the current task
            task = (
                self.api_resource.tasks().get(tasklist=tasklist, task=task_id).execute()
            )

            # Update only the fields that were provided
            if title is not None:
                task["title"] = title

            if notes is not None:
                task["notes"] = notes

            if status is not None:
                task["status"] = status

            if due is not None:
                task["due"] = due

            # Update the task
            result = (
                self.api_resource.tasks()
                .update(tasklist=tasklist, task=task_id, body=task)
                .execute()
            )

            task_id = result.get("id", "Unknown")
            task_title = result.get("title", "Unknown")
            response = f"Task updated successfully!\nID: {task_id}\nTitle: {task_title}"

            if result.get("status"):
                response += f"\nStatus: {result['status']}"

            if result.get("notes"):
                response += f"\nNotes: {result['notes']}"

            if result.get("due"):
                response += f"\nDue: {result['due']}"

            return response

        except Exception as e:
            return f"An error occurred while updating the task: {str(e)}"
