"""Google Tasks Toolkit."""

from __future__ import annotations

from typing import TYPE_CHECKING, List

from langchain_core.tools import BaseTool, BaseToolkit
from pydantic import ConfigDict, Field

from langchain_google_community.tasks.create_task import TasksCreateTask
from langchain_google_community.tasks.delete_task import TasksDeleteTask
from langchain_google_community.tasks.get_task import TasksGetTask
from langchain_google_community.tasks.list_tasks import TasksListTasks
from langchain_google_community.tasks.update_task import TasksUpdateTask
from langchain_google_community.tasks.utils import build_tasks_service

if TYPE_CHECKING:
    # This is for linting and IDE typehints
    from googleapiclient.discovery import Resource  # type: ignore[import]
else:
    try:
        # We do this so pydantic can resolve the types when instantiating
        from googleapiclient.discovery import Resource
    except ImportError:
        pass


class TasksToolkit(BaseToolkit):
    """Toolkit for interacting with Google Tasks.

    This toolkit provides tools for creating, listing, updating,
    deleting, and retrieving tasks from Google Tasks.

    Setup:
        Install ``langchain-google-community`` and set up Google authentication.

        .. code-block:: bash

            pip install -U langchain-google-community

        You'll need to enable the Google Tasks API and set up credentials.

    Instantiation:
        .. code-block:: python

            from langchain_google_community import TasksToolkit

            toolkit = TasksToolkit()

    Tools:
        .. code-block:: python

            tools = toolkit.get_tools()
            # Returns: [TasksCreateTask, TasksListTasks, TasksUpdateTask,
            #           TasksDeleteTask, TasksGetTask]

    Use within an agent:
        .. code-block:: python

            from langchain_openai import ChatOpenAI
            from langgraph.prebuilt import create_react_agent

            llm = ChatOpenAI(model="gpt-4o-mini")
            agent_executor = create_react_agent(llm, tools)

            example_query = "Create a task to review the quarterly report"
            events = agent_executor.stream(
                {"messages": [("user", example_query)]},
                stream_mode="values",
            )
            for event in events:
                event["messages"][-1].pretty_print()

    !!! warning "Security"
        This toolkit contains tools that can read and modify the state of a
        service. For example, it can create, update, and delete tasks
        on behalf of the associated account.

        See [Security Policy](https://docs.langchain.com/oss/python/security-policy)
        for more information.
    """

    api_resource: Resource = Field(default_factory=build_tasks_service)

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    def get_tools(self) -> List[BaseTool]:
        """Get the tools in the toolkit.

        Returns:
            A list of tools for interacting with Google Tasks.
        """
        return [
            TasksCreateTask(api_resource=self.api_resource),
            TasksListTasks(api_resource=self.api_resource),
            TasksUpdateTask(api_resource=self.api_resource),
            TasksDeleteTask(api_resource=self.api_resource),
            TasksGetTask(api_resource=self.api_resource),
        ]
