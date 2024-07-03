""" Excepetions module
"""


class WrongTensorShapeException(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)
