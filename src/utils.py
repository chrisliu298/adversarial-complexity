from typing import List


def format_output(output: List) -> str:
    """Print a dictionary output in a predefined format"""
    output_string = ""
    for o in output:
        for k, v in o.items():
            output_string += str(k)
            output_string += "\t"
            output_string += "\t".join(str(i) for i in v)
        output_string += "\n"
    return output_string
