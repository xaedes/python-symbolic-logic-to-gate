
import re

def split_name_id(string, default_id=0, infix="_", to_int=True):
    string = str(string)
    match = re.match(r"^([^\d]*)(|\d+)$", string)
    if match is None: return string, default_id
    int_or = lambda s, default_value: (
        s      if not to_int else
        int(s) if len(s) > 0 else
        default_value
    )
    name = match.group(1)
    id = int_or(match.group(2), default_id)
    if name.endswith(infix): name = name[:-len(infix)]
    return (name, id)
