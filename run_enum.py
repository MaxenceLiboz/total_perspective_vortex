from enum import Enum

class Run(Enum):
    MOTOR_LEFT_RIGHT = [3, 7, 11]
    IMAGERY_LEFT_RIGHT = [4, 8, 12]
    MOTOR_HANDS_FEET = [5, 9, 13]
    IMAGERY_HANDS_FEET = [6, 10, 14]


    def get_by_index(index):
        return list(Run)[index - 1].value
    

    def get_all():
        return [Run.MOTOR_LEFT_RIGHT.value, Run.IMAGERY_LEFT_RIGHT.value, Run.MOTOR_HANDS_FEET.value, Run.IMAGERY_HANDS_FEET.value]