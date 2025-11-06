"""Tool for creating new Google Spreadsheets."""

from typing import Any, Dict, List, Optional, Type

from langchain_core.callbacks import CallbackManagerForToolRun
from pydantic import BaseModel, Field

from .base import SheetsBaseTool
from .utils import validate_range_name

# ============================================================================
# 1. CREATE SPREADSHEET SCHEMA
# ============================================================================


class CreateSpreadsheetSchema(BaseModel):
    """Input schema for `SheetsCreateSpreadsheetTool`."""

    title: str = Field(description="The title of the new spreadsheet.")
    locale: Optional[str] = Field(
        default="en_US",
        description=(
            "The locale of the spreadsheet in one of the ISO 639-1 "
            "language codes (e.g. 'en_US', 'fr_FR'). Defaults to 'en_US'."
        ),
    )
    time_zone: Optional[str] = Field(
        default="America/New_York",
        description=(
            "The time zone of the spreadsheet, specified as a time zone ID "
            "from the IANA Time Zone Database (e.g. 'America/New_York', "
            "'Europe/London'). Defaults to 'America/New_York'."
        ),
    )
    auto_recalc: Optional[str] = Field(
        default="ON_CHANGE",
        description=(
            "The amount of time to wait before volatile functions are "
            "recalculated. Options: 'ON_CHANGE', 'MINUTE', 'HOUR'. "
            "Defaults to 'ON_CHANGE'."
        ),
    )
    initial_data: Optional[List[List[Any]]] = Field(
        default=None,
        description=(
            "Optional initial data to populate the spreadsheet. "
            "2D array where each inner array represents a row. "
            "Supports strings, numbers, booleans, and formulas."
        ),
    )
    initial_range: Optional[str] = Field(
        default="A1",
        description="The range where initial data should be placed. Defaults to 'A1'.",
    )


# ============================================================================
# 2. CREATE SPREADSHEET TOOL
# ============================================================================


class SheetsCreateSpreadsheetTool(SheetsBaseTool):
    """Tool for creating new Google Spreadsheets.

    Inherits from
    [`SheetsBaseTool`][langchain_google_community.sheets.base.SheetsBaseTool].

    Creates spreadsheets with configurable properties and optional initial data.

    !!! note "Authentication Required"
        Requires OAuth2 authentication. Use `api_resource` parameter with
        authenticated Google Sheets service.

    Tool Output:
        success (bool): Whether operation succeeded.
        spreadsheet_id (str): Unique ID of the created spreadsheet.
        spreadsheet_url (str): Direct URL to open the spreadsheet.
        title (str): The spreadsheet title.
        locale (str): The locale setting.
        time_zone (str): The timezone setting.
        auto_recalc (str): The recalculation setting.
        created (bool): Whether creation succeeded.
        initial_data_added (bool): Whether initial data was added (if provided).
        initial_data_cells_updated (int): Number of cells updated (if data added).
        initial_data_range (str): Range where data was placed (if data added).

    ???+ example "Basic Usage"

        Create a simple spreadsheet:

        ```python
        from langchain_google_community.sheets import SheetsCreateSpreadsheetTool

        tool = SheetsCreateSpreadsheetTool(api_resource=service)
        result = tool.run({"title": "My New Spreadsheet"})
        print(f"Created: {result['spreadsheet_url']}")
        ```


    ??? example "With Initial Data"

        Create spreadsheet with pre-populated data:

        ```python
        result = tool.run(
            {
                "title": "Sales Report",
                "initial_data": [
                    ["Name", "Region", "Sales"],
                    ["Alice", "East", "95000"],
                    ["Bob", "West", "87000"],
                ],
                "initial_range": "A1",
            }
        )
        ```


    ??? example "Custom Configuration"

        Create with locale and timezone settings:

        ```python
        result = tool.run(
            {
                "title": "European Report",
                "locale": "fr_FR",
                "time_zone": "Europe/Paris",
                "auto_recalc": "HOUR",
            }
        )
        ```

    Raises:
        ValueError: If write permissions unavailable or validation fails.
        Exception: If API call fails.
    """

    name: str = "sheets_create_spreadsheet"
    description: str = (
        "Create a new Google Spreadsheet with the specified title and properties. "
        "Can optionally populate the spreadsheet with initial data."
    )
    args_schema: Type[BaseModel] = CreateSpreadsheetSchema

    def _run(
        self,
        title: str,
        locale: Optional[str] = "en_US",
        time_zone: Optional[str] = "America/New_York",
        auto_recalc: Optional[str] = "ON_CHANGE",
        initial_data: Optional[List[List[Any]]] = None,
        initial_range: Optional[str] = "A1",
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Dict[str, Any]:
        """Create a new Google Spreadsheet.

        Args:
            title: Title of the new spreadsheet.
            locale: Locale of the spreadsheet.
            time_zone: Timezone of the spreadsheet.
            auto_recalc: Recalculation setting.
            initial_data: Optional 2D array of initial data to populate.
            initial_range: Range where initial data should be placed.
            run_manager: Optional callback manager.

        Returns:
            success (bool): Whether operation succeeded.
            spreadsheet_id (str): Unique ID of created spreadsheet.
            spreadsheet_url (str): Direct URL to spreadsheet.
            title (str): Spreadsheet title.
            locale (str): Locale setting.
            time_zone (str): Timezone setting.
            auto_recalc (str): Recalculation setting.
            created (bool): Whether creation succeeded.
            initial_data_added (bool): Whether initial data was added.
            initial_data_cells_updated (int): Number of cells updated.
            initial_data_range (str): Range where data was placed.

        Raises:
            ValueError: If write permissions unavailable or validation fails.
            Exception: If API call fails.
        """
        # Check write permissions (requires OAuth2)
        self._check_write_permissions()

        try:
            # Get the Google Sheets service
            service = self._get_service()

            # Build the spreadsheet properties
            spreadsheet_body = {
                "properties": {
                    "title": title,
                    "locale": locale,
                    "timeZone": time_zone,
                    "autoRecalc": auto_recalc,
                }
            }

            # Create the spreadsheet
            spreadsheet = (
                service.spreadsheets()
                .create(
                    body=spreadsheet_body,
                    fields="spreadsheetId,spreadsheetUrl,properties",
                )
                .execute()
            )

            spreadsheet_id = spreadsheet.get("spreadsheetId")
            spreadsheet_url = spreadsheet.get("spreadsheetUrl")
            properties = spreadsheet.get("properties", {})

            result = {
                "success": True,
                "spreadsheet_id": spreadsheet_id,
                "spreadsheet_url": spreadsheet_url,
                "title": properties.get("title", title),
                "locale": properties.get("locale", locale),
                "time_zone": properties.get("timeZone", time_zone),
                "auto_recalc": properties.get("autoRecalc", auto_recalc),
                "created": True,
            }

            # Add initial data if provided
            if initial_data:
                try:
                    # Validate initial_range before making API call
                    validated_range = validate_range_name(initial_range or "A1")

                    body = {"values": initial_data}

                    update_result = (
                        service.spreadsheets()
                        .values()
                        .update(
                            spreadsheetId=spreadsheet_id,
                            range=validated_range,
                            valueInputOption="RAW",
                            body=body,
                        )
                        .execute()
                    )

                    result["initial_data_added"] = True
                    result["initial_data_cells_updated"] = update_result.get(
                        "updatedCells", 0
                    )
                    result["initial_data_range"] = initial_range

                except Exception as e:
                    # If initial data fails, still return the created spreadsheet
                    result["initial_data_error"] = str(e)
                    result["initial_data_added"] = False

            return result

        except Exception as error:
            raise Exception(f"Error creating spreadsheet: {error}") from error
