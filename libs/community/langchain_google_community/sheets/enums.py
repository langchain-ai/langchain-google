"""Google Sheets API enums and constants.

This module contains all the enums and constants used across the Google Sheets
module for type safety, validation, and better developer experience.
"""

from enum import Enum


class ValueRenderOption(str, Enum):
    """Google Sheets value render options.

    Determines how values should be rendered in the response.
    """

    FORMATTED_VALUE = "FORMATTED_VALUE"
    """Values will be calculated and formatted in the reply according to the 
    cell's formatting. Formatting is based on the spreadsheet's locale, not 
    the requesting user's locale."""

    UNFORMATTED_VALUE = "UNFORMATTED_VALUE"
    """Values will be calculated, but not formatted in the reply. For example, 
    if A1 is 1.23 and A2 is =A1 and formatted as currency, then A2 would 
    return the number 1.23, not "$1.23"."""

    FORMULA = "FORMULA"
    """Values will not be calculated. The reply will include the formulas. 
    For example, if A1 is 1.23 and A2 is =A1 and formatted as currency, 
    then A2 would return "=A1"."""


class DateTimeRenderOption(str, Enum):
    """Google Sheets date/time render options.

    Determines how dates, times, and durations should be rendered in the response.
    """

    SERIAL_NUMBER = "SERIAL_NUMBER"
    """Instructs date, time, datetime, and duration fields to be output as 
    doubles in "serial number" format, as popularized by Lotus 1-2-3. The 
    whole number portion of the value (counting the epoch of 1900-01-01 as 1) 
    is the number of days since 1900-01-01. The fractional portion of the 
    value counts the time as a fraction of the day."""

    FORMATTED_STRING = "FORMATTED_STRING"
    """Instructs date, time, datetime, and duration fields to be output as 
    strings in their given number format (which depends on the spreadsheet 
    locale)."""


class MajorDimension(str, Enum):
    """Google Sheets major dimension options.

    Determines how the values should be oriented in the response.
    """

    ROWS = "ROWS"
    """The values are returned as a 2D array with rows as the major dimension.
    Each row contains all the values for that row."""

    COLUMNS = "COLUMNS"
    """The values are returned as a 2D array with columns as the major dimension.
    Each column contains all the values for that column."""


# Common constants for validation and default values
DEFAULT_VALUE_RENDER_OPTION = ValueRenderOption.FORMATTED_VALUE
DEFAULT_DATE_TIME_RENDER_OPTION = DateTimeRenderOption.SERIAL_NUMBER
DEFAULT_MAJOR_DIMENSION = MajorDimension.ROWS

# Valid options for validation
VALID_VALUE_RENDER_OPTIONS = [option.value for option in ValueRenderOption]
VALID_DATE_TIME_RENDER_OPTIONS = [option.value for option in DateTimeRenderOption]
VALID_MAJOR_DIMENSIONS = [option.value for option in MajorDimension]


class FilterConditionType(str, Enum):
    """Filter condition types for Google Sheets data filters."""

    # Number conditions
    NUMBER_GREATER = "NUMBER_GREATER"
    """Number is greater than the specified value."""
    NUMBER_GREATER_THAN_OR_EQUAL = "NUMBER_GREATER_THAN_OR_EQUAL"
    """Number is greater than or equal to the specified value."""
    NUMBER_LESS = "NUMBER_LESS"
    """Number is less than the specified value."""
    NUMBER_LESS_THAN_OR_EQUAL = "NUMBER_LESS_THAN_OR_EQUAL"
    """Number is less than or equal to the specified value."""
    NUMBER_EQUAL = "NUMBER_EQUAL"
    """Number is equal to the specified value."""
    NUMBER_NOT_EQUAL = "NUMBER_NOT_EQUAL"
    """Number is not equal to the specified value."""
    NUMBER_BETWEEN = "NUMBER_BETWEEN"
    """Number is between two specified values."""

    # Text conditions
    TEXT_CONTAINS = "TEXT_CONTAINS"
    """Text contains the specified value."""
    TEXT_DOES_NOT_CONTAIN = "TEXT_DOES_NOT_CONTAIN"
    """Text does not contain the specified value."""
    TEXT_STARTS_WITH = "TEXT_STARTS_WITH"
    """Text starts with the specified value."""
    TEXT_ENDS_WITH = "TEXT_ENDS_WITH"
    """Text ends with the specified value."""
    TEXT_EQUAL = "TEXT_EQUAL"
    """Text is equal to the specified value."""
    TEXT_NOT_EQUAL = "TEXT_NOT_EQUAL"
    """Text is not equal to the specified value."""

    # Date conditions
    DATE_IS_AFTER = "DATE_IS_AFTER"
    """Date is after the specified value."""
    DATE_IS_BEFORE = "DATE_IS_BEFORE"
    """Date is before the specified value."""
    DATE_IS_ON_OR_AFTER = "DATE_IS_ON_OR_AFTER"
    """Date is on or after the specified value."""
    DATE_IS_ON_OR_BEFORE = "DATE_IS_ON_OR_BEFORE"
    """Date is on or before the specified value."""
    DATE_IS_BETWEEN = "DATE_IS_BETWEEN"
    """Date is between two specified values."""
    DATE_IS_EQUAL = "DATE_IS_EQUAL"
    """Date is equal to the specified value."""

    # Boolean conditions
    BOOLEAN_IS_TRUE = "BOOLEAN_IS_TRUE"
    """Boolean is true."""
    BOOLEAN_IS_FALSE = "BOOLEAN_IS_FALSE"
    """Boolean is false."""


# Default values and validation lists for filter conditions
DEFAULT_FILTER_CONDITION_TYPE = FilterConditionType.TEXT_CONTAINS
VALID_FILTER_CONDITION_TYPES = [option.value for option in FilterConditionType]

# Grouped filter condition types for easier validation
NUMBER_CONDITIONS = [
    FilterConditionType.NUMBER_GREATER.value,
    FilterConditionType.NUMBER_GREATER_THAN_OR_EQUAL.value,
    FilterConditionType.NUMBER_LESS.value,
    FilterConditionType.NUMBER_LESS_THAN_OR_EQUAL.value,
    FilterConditionType.NUMBER_EQUAL.value,
    FilterConditionType.NUMBER_NOT_EQUAL.value,
    FilterConditionType.NUMBER_BETWEEN.value,
]

TEXT_CONDITIONS = [
    FilterConditionType.TEXT_CONTAINS.value,
    FilterConditionType.TEXT_DOES_NOT_CONTAIN.value,
    FilterConditionType.TEXT_STARTS_WITH.value,
    FilterConditionType.TEXT_ENDS_WITH.value,
    FilterConditionType.TEXT_EQUAL.value,
    FilterConditionType.TEXT_NOT_EQUAL.value,
]

DATE_CONDITIONS = [
    FilterConditionType.DATE_IS_AFTER.value,
    FilterConditionType.DATE_IS_BEFORE.value,
    FilterConditionType.DATE_IS_ON_OR_AFTER.value,
    FilterConditionType.DATE_IS_ON_OR_BEFORE.value,
    FilterConditionType.DATE_IS_BETWEEN.value,
    FilterConditionType.DATE_IS_EQUAL.value,
]

BOOLEAN_CONDITIONS = [
    FilterConditionType.BOOLEAN_IS_TRUE.value,
    FilterConditionType.BOOLEAN_IS_FALSE.value,
]


class ValueInputOption(str, Enum):
    """Google Sheets value input options for write operations.

    Determines how input values should be interpreted when writing to cells.
    """

    RAW = "RAW"
    """Values are stored exactly as provided without any parsing.
    For example, "=1+2" will be stored as the literal string "=1+2"."""

    USER_ENTERED = "USER_ENTERED"
    """Values are parsed as if the user typed them into the UI.
    For example, "=1+2" will be parsed as a formula and display "3".
    Numbers, dates, and formulas are automatically detected and parsed."""


class InsertDataOption(str, Enum):
    """Google Sheets insert data options for append operations.

    Determines how data should be inserted when appending to a table.
    """

    OVERWRITE = "OVERWRITE"
    """Data overwrites existing data after the table. This is the default."""

    INSERT_ROWS = "INSERT_ROWS"
    """Rows are inserted for the new data. Existing data is shifted down."""


# Default values for write operations
DEFAULT_VALUE_INPUT_OPTION = ValueInputOption.USER_ENTERED
DEFAULT_INSERT_DATA_OPTION = InsertDataOption.OVERWRITE

# Valid options for validation
VALID_VALUE_INPUT_OPTIONS = [option.value for option in ValueInputOption]
VALID_INSERT_DATA_OPTIONS = [option.value for option in InsertDataOption]
