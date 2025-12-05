"""Create a task in Google Tasks."""

from typing import Optional, Type

from langchain_core.callbacks import CallbackManagerForToolRun
from pydantic import BaseModel, Field

from langchain_google_community.tasks.base import TasksBaseTool


class CreateTaskSchema(BaseModel):
    """Input schema for `TasksCreateTask`."""

    title: str = Field(..., description="The title of the task.")

    notes: Optional[str] = Field(
        default=None, description="Notes or description for the task."
    )

    due: Optional[str] = Field(
        default=None,
        description=(
            "Due date of the task in RFC 3339 format (e.g., '2024-12-31T23:59:59Z'). "
            "Optional."
        ),
    )

    tasklist: str = Field(
        default="@default",
        description=(
            "The task list ID to create the task in. "
            "Use '@default' for the default task list."
        ),
    )


class TasksCreateTask(TasksBaseTool):  # type: ignore[override]
    """Tool that creates a task in Google Tasks."""

    name: str = "create_google_task"

    description: str = (
        "Use this tool to create a new task in Google Tasks. "
        "The input must include the title of the task. "
        "You can optionally include notes and a due date."
    )

    args_schema: Type[CreateTaskSchema] = CreateTaskSchema

    def _run(
        self,
        title: str,
        notes: Optional[str] = None,
        due: Optional[str] = None,
        tasklist: str = "@default",
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Create a task in Google Tasks.

        Args:
            title: The title of the task.
            notes: Optional notes or description for the task.
            due: Optional due date in RFC 3339 format.
            tasklist: The task list ID. Defaults to '@default'.
            run_manager: Optional callback manager.

        Returns:
            A string confirming the task creation with task details.
        """
        try:
            task_body = {"title": title}

            if notes:
                task_body["notes"] = notes

            if due:
                task_body["due"] = due

            result = (
                self.api_resource.tasks()
                .insert(tasklist=tasklist, body=task_body)
                .execute()
            )

            task_id = result.get("id", "Unknown")
            task_title = result.get("title", "Unknown")

            response = f"Task created successfully!\nID: {task_id}\nTitle: {task_title}"

            if result.get("notes"):
                response += f"\nNotes: {result['notes']}"

            if result.get("due"):
                response += f"\nDue: {result['due']}"

            return response

        except Exception as e:
            return f"An error occurred while creating the task: {str(e)}"
