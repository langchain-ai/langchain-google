"""Create spreadsheet tool for Google Sheets.

This module contains the tool for creating new Google Spreadsheets with
configurable properties and initial data.

Note: Requires OAuth2 authentication (api_resource).
"""

from typing import Any, Dict, List, Optional, Type

from langchain_core.callbacks import CallbackManagerForToolRun
from pydantic import BaseModel, Field

from .base import SheetsBaseTool
from .utils import validate_range_name

# ============================================================================
# 1. CREATE SPREADSHEET SCHEMA
# ============================================================================


class CreateSpreadsheetSchema(BaseModel):
    """Schema for creating a new Google Spreadsheet."""

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
    """Tool for creating a new Google Spreadsheet.

    This tool creates a new spreadsheet with configurable properties and
    optional initial data. Perfect for dynamically generating reports,
    creating data collection forms, or initializing new project workspaces
    with pre-populated templates.

    Instantiate:
        .. code-block:: python

            from langchain_google_community.sheets import SheetsCreateSpreadsheetTool

            tool = SheetsCreateSpreadsheetTool(api_resource=service)

    Invoke directly:
        .. code-block:: python

            result = tool.run(
                {
                    "title": "Test Spreadsheet - Full Options",
                    "locale": "en_US",
                    "time_zone": "America/Los_Angeles",
                    "auto_recalc": "ON_CHANGE",
                    "initial_data": [
                        ["Name", "Age", "City", "Score"],
                        ["Alice", "25", "New York", "95"],
                        ["Bob", "30", "San Francisco", "87"],
                        ["Charlie", "28", "Chicago", "92"],
                    ],
                    "initial_range": "A1",
                }
            )

    Invoke with agent:
        .. code-block:: python

            agent.invoke({"input": "Create a new employee tracking spreadsheet"})

    Returns:
        Dictionary containing:
            - success (bool): Always True for successful operations
            - spreadsheet_id (str): The unique ID of the created spreadsheet
            - spreadsheet_url (str): Direct URL to open the spreadsheet
            - title (str): The spreadsheet title
            - locale (str): The locale setting
            - time_zone (str): The timezone setting
            - auto_recalc (str): The recalculation setting
            - created (bool): Whether creation succeeded
            - initial_data_added (bool): Whether initial data was added
                (if initial_data provided)
            - initial_data_cells_updated (int): Number of cells populated
                (if data added)
            - initial_data_range (str): Where data was placed (if data added)

    Configuration Options:
        - title: Required - The spreadsheet name
        - locale: ISO 639-1 language code (e.g., 'en_US', 'fr_FR')
        - time_zone: IANA timezone (e.g., 'America/New_York', 'Europe/London')
        - auto_recalc: 'ON_CHANGE', 'MINUTE', or 'HOUR'
        - initial_data: 2D array for pre-populating cells
        - initial_range: Where to place initial data (default: 'A1')

    Example Response:
        {
            "success": True,
            "spreadsheet_id": "1TI6vO9eGsAeXcfgEjoEYcu4RgSZCUF4vdWGLBpg9-fg",
            "spreadsheet_url": "https://docs.google.com/spreadsheets/d/1TI6vO9eGsAeXcfgEjoEYcu4RgSZCUF4vdWGLBpg9-fg/edit",
            "title": "Test Spreadsheet - Full Options",
            "locale": "en_US",
            "time_zone": "America/Los_Angeles",
            "auto_recalc": "ON_CHANGE",
            "created": True,
            "initial_data_added": True,
            "initial_data_cells_updated": 16,
            "initial_data_range": "A1"
        }

    Raises:
        Exception: If authentication fails, quota is exceeded, or API errors occur
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
            title: The title of the new spreadsheet.
            locale: The locale of the spreadsheet (e.g. 'en_US', 'fr_FR').
            time_zone: The time zone of the spreadsheet (e.g. 'America/New_York').
            auto_recalc: The recalculation setting ('ON_CHANGE', 'MINUTE', 'HOUR').
            initial_data: Optional 2D array of initial data to populate.
            initial_range: The range where initial data should be placed.
            run_manager: Optional callback manager.

        Returns:
            Dict containing the spreadsheet ID, URL, and creation details.

        Raises:
            ValueError: If write permissions are not available or validation fails.
            Exception: If the API call fails.
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
