"""Record conversion through the public Sheets read tools."""

from typing import Any
from unittest.mock import MagicMock

import pytest

from langchain_google_community.sheets import (
    SheetsBatchReadDataTool,
    SheetsFilteredReadDataTool,
    SheetsReadDataTool,
)

SPREADSHEET_ID = "1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms"


def _read_values(
    mode: str, values: list[list[str]], records: bool, numeric: bool
) -> list[Any]:
    service = MagicMock()
    args: dict[str, Any] = {
        "spreadsheet_id": SPREADSHEET_ID,
        "convert_to_records": records,
        "numericise_values": numeric,
    }
    if mode == "single":
        service.spreadsheets().values().get().execute.return_value = {"values": values}
        tool = SheetsReadDataTool.model_construct(api_resource=service)
        return tool.invoke({**args, "range_name": "A1:B2"})["values"]
    if mode == "batch":
        service.spreadsheets().values().batchGet().execute.return_value = {
            "valueRanges": [{"range": "A1:B2", "values": values}]
        }
        batch_tool = SheetsBatchReadDataTool.model_construct(api_resource=service)
        return batch_tool.invoke({**args, "ranges": ["A1:B2"]})["results"][0]["data"]

    service.spreadsheets().getByDataFilter().execute.return_value = {
        "spreadsheetId": SPREADSHEET_ID,
        "sheets": [
            {
                "data": [
                    {
                        "rowData": [
                            {"values": [{"formattedValue": cell} for cell in row]}
                            for row in values
                        ]
                    }
                ]
            }
        ],
    }
    filtered_tool = SheetsFilteredReadDataTool.model_construct(api_resource=service)
    result = filtered_tool.invoke({**args, "data_filters": [{"a1Range": "A1:B2"}]})
    return result["sheets"][0]["data"][0]


@pytest.mark.parametrize("mode", ["single", "batch", "filtered"])
@pytest.mark.parametrize("numeric", [False, True])
@pytest.mark.parametrize("values", [[], [["Name", "Age"]]])
def test_no_data_rows_produce_empty_records(
    mode: str, numeric: bool, values: list[list[str]]
) -> None:
    assert _read_values(mode, values, records=True, numeric=numeric) == []


@pytest.mark.parametrize("mode", ["single", "batch", "filtered"])
@pytest.mark.parametrize("numeric", [False, True])
def test_header_only_array_mode_preserves_headers(mode: str, numeric: bool) -> None:
    values = [["Name", "Age"]]
    assert _read_values(mode, values, records=False, numeric=numeric) == values


@pytest.mark.parametrize("mode", ["single", "batch", "filtered"])
@pytest.mark.parametrize("numeric", [False, True])
def test_records_still_convert_data_rows(mode: str, numeric: bool) -> None:
    assert _read_values(
        mode, [["Name", "Age"], ["Ada", "12"]], records=True, numeric=numeric
    ) == [{"Name": "Ada", "Age": 12 if numeric else "12"}]
