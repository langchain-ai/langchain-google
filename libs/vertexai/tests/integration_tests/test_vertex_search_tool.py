import os

import pytest

from langchain_google_vertexai.tools import VertexSearchTool


@pytest.mark.extended
def test_vertex_search_tool():

    project_id = os.environ["PROJECT_ID"]
    engine_id = os.environ["VERTEX_SEARCH_APP_ID"]
    
    tool = VertexSearchTool(
        name="vertex-search", 
        description="Vertex Search Tool", 
        engine_id=engine_id,
        project_id=project_id
    )

    response = tool.run("How many Champion's Leagues has Real Madrid won?")

    assert(isinstance(response, str))