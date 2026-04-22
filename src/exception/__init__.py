import sys


class MyException(Exception):
    """Custom exception for AutoML pipeline errors."""

    def __init__(self, error_message: str, error_detail: sys = sys):
        super().__init__(error_message)
        self.error_message = self._format_error(error_message, error_detail)

    @staticmethod
    def _format_error(message: str, error_detail: sys) -> str:
        _, _, exc_tb = error_detail.exc_info()
        if exc_tb is not None:
            file_name = exc_tb.tb_frame.f_code.co_filename
            line_number = exc_tb.tb_lineno
            return f"Error in [{file_name}] at line [{line_number}]: {message}"
        return f"Error: {message}"

    def __str__(self):
        return self.error_message