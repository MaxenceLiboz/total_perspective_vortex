from enum import Enum

class Run(Enum):
    MOTOR_LEFT_RIGHT = [3, 7, 11]
    IMAGERY_LEFT_RIGHT = [4, 8, 12]
    MOTOR_HANDS_FEET = [5, 9, 13]
    IMAGERY_HANDS_FEET = [6, 10, 14]
    MOTOR_IMAGERY_LEFT_RIGHT = [3, 4, 7, 8, 11, 12]
    MOTOR_IMAGERY_HANDS_FEET = [5, 6, 9, 10, 13, 14]


    def get_by_index(index):
        try:
            return list(Run)[index - 1]
        except:
            raise Exception(f'Index {index} not found in Run enum')
    

    def get_all():
        return [run for run in Run]
    

    def __str__():
        str = ""
        for index, run in enumerate(Run):
            str += f'{index + 1}: {run.name} '
        return str