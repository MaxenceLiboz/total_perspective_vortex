from enum import Enum

class Run(Enum):
    MOTOR_LEFT_RIGHT = [3, 7, 11]
    IMAGERY_LEFT_RIGHT = [4, 8, 12]
    MOTOR_HANDS_FEET = [5, 9, 13]
    IMAGERY_HANDS_FEET = [6, 10, 14]
    MOTOR_IMAGERY_LEFT_RIGHT = [3, 4, 7, 8, 11, 12]
    MOTOR_IMAGERY_HANDS_FEET = [5, 6, 9, 10, 13, 14]

    def get_by_index(index):
        return list(Run)[index - 1].value
    
    def get_all():
        return [run.value for run in Run]