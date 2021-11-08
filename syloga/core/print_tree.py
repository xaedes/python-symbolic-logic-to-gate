
from syloga.ast.Expression import Expression

def print_tree(expression, print_result = True, max_item_string_length = 40):
    stack = [(0,expression)]
    lines = []
    single_indent = "  "
    while len(stack) > 0:
        depth, item = stack.pop()
        is_expression = isinstance(item, Expression)
        string = type(item).__name__
        #string += "(" + str(id(type(item) ))+ ")"
        string += ": " 
        #if not is_expression:
        str_item = str(item)
        if len(str_item) > max_item_string_length:
            str_item = str_item[:max_item_string_length][:-3]+"..."
        string += str_item
        prefix = single_indent * depth
        line = prefix + string
        #print(line)
        lines.append(line)
        child_idx = 0
        
        if is_expression:
            for arg in reversed(item.args):
                stack.append((depth+1,arg))
        elif type(item) != str:
            try:
                args = list(iter(item))
                for arg in reversed(args):
                    stack.append((depth+1,arg))
            except TypeError:
                pass

    result = "\n".join(lines)
    if print_result:
        print(result)
    return result

