"""Google Sheets API URLs and endpoints for read-only operations.

This module contains URL patterns and endpoints for Google Sheets API v4
read-only operations. These URLs are used for reading spreadsheet data
without modification capabilities.
"""

# Base URLs
SPREADSHEETS_API_V4_BASE_URL: str = "https://sheets.googleapis.com/v4/spreadsheets"

# Read-only Spreadsheet URLs
SPREADSHEET_URL: str = SPREADSHEETS_API_V4_BASE_URL + "/%s"
SPREADSHEET_GET_BY_DATA_FILTER_URL: str = (
    SPREADSHEETS_API_V4_BASE_URL + "/%s:getByDataFilter"
)

# Read-only Values URLs
SPREADSHEET_VALUES_URL: str = SPREADSHEETS_API_V4_BASE_URL + "/%s/values/%s"
SPREADSHEET_VALUES_BATCH_URL: str = SPREADSHEETS_API_V4_BASE_URL + "/%s/values:batchGet"
