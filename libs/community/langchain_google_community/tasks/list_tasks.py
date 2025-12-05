"""List tasks from Google Tasks."""

from typing import Optional, Type

from langchain_core.callbacks import CallbackManagerForToolRun
from pydantic import BaseModel, Field

from langchain_google_community.tasks.base import TasksBaseTool


class ListTasksSchema(BaseModel):
    """Input schema for `TasksListTasks`."""

    tasklist: str = Field(
        default="@default",
        description=(
            "The task list ID to retrieve tasks from. "
            "Use '@default' for the default task list."
        ),
    )

    max_results: int = Field(
        default=10,
        description="Maximum number of tasks to return. Default is 10.",
    )

    show_completed: bool = Field(
        default=False,
        description=(
            "Whether to include completed tasks in the results. Default is False."
        ),
    )

    show_hidden: bool = Field(
        default=False,
        description="Whether to include hidden tasks in the results. Default is False.",
    )


class TasksListTasks(TasksBaseTool):  # type: ignore[override]
    """Tool that lists tasks from Google Tasks."""

    name: str = "list_google_tasks"

    description: str = (
        "Use this tool to list tasks from a Google Tasks list. "
        "You can specify the task list ID, maximum number of results, "
        "and whether to include completed or hidden tasks."
    )

    args_schema: Type[ListTasksSchema] = ListTasksSchema

    def _run(
        self,
        tasklist: str = "@default",
        max_results: int = 10,
        show_completed: bool = False,
        show_hidden: bool = False,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """List tasks from Google Tasks.

        Args:
            tasklist: The task list ID. Defaults to '@default'.
            max_results: Maximum number of tasks to return.
            show_completed: Whether to include completed tasks.
            show_hidden: Whether to include hidden tasks.
            run_manager: Optional callback manager.

        Returns:
            A formatted string with the list of tasks.
        """
        try:
            results = (
                self.api_resource.tasks()
                .list(
                    tasklist=tasklist,
                    maxResults=max_results,
                    showCompleted=show_completed,
                    showHidden=show_hidden,
                )
                .execute()
            )

            tasks = results.get("items", [])

            if not tasks:
                return "No tasks found."

            response = f"Found {len(tasks)} task(s):\n\n"

            for i, task in enumerate(tasks, 1):
                task_id = task.get("id", "Unknown")
                title = task.get("title", "No title")
                status = task.get("status", "Unknown")

                response += f"{i}. {title}\n"
                response += f"   ID: {task_id}\n"
                response += f"   Status: {status}\n"

                if task.get("notes"):
                    notes = task["notes"][:100]  # Truncate long notes
                    response += (
                        f"   Notes: {notes}...\n"
                        if len(task["notes"]) > 100
                        else f"   Notes: {notes}\n"
                    )

                if task.get("due"):
                    response += f"   Due: {task['due']}\n"

                response += "\n"

            return response.strip()

        except Exception as e:
            return f"An error occurred while listing tasks: {str(e)}"
