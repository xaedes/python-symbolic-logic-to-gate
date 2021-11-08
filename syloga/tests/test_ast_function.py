
import syloga.ast.core as core
import syloga.ast.containers as containers
import syloga.ast.function as function

def assert_copy_test(a):
    a_copy = a.copy()
    assert(a is not a_copy)
    assert(a == a_copy)

def test_FunctionArgument():
    in_a = function.FunctionArgument("a")
    for in_a in [
                function.FunctionArgument.In("a"),
                function.FunctionArgument("a",is_in=True),
                function.FunctionArgument("a"),
            ]:

        assert(in_a.name == "a")
        assert(in_a.is_in == True)
        assert(in_a.is_out == False)
        assert_copy_test(in_a)

    for out_a in [
                function.FunctionArgument.Out("a"),
                function.FunctionArgument("a",is_in=False,is_out=True),
            ]:

        assert(out_a.name == "a")
        assert(out_a.is_in == False)
        assert(out_a.is_out == True)
        assert_copy_test(out_a)

def test_FunctionDeclaration():
    f = function.FunctionDeclaration("f", [])
    assert(f.name == "f")
    assert(len(f.arguments) == 0)
    assert(f.list_dependees() == [])
    assert(f.list_dependencies() == [])
    assert_copy_test(f)
    
    f_call = f.call()
    assert(f_call.function_declaration == f)
    
    In = function.FunctionArgument.In
    f_ab = function.FunctionDeclaration("f", [In("a"),In("b"),])
    assert(f_ab.name == "f")
    assert(len(f_ab.arguments) == 2)
    assert(f_ab.list_dependees() == [])
    assert(f_ab.list_dependencies() == [In("a"), In("b")])
    assert_copy_test(f_ab)

    f_ab_call = f_ab.call()
    assert(f_ab_call.function_declaration == f_ab)
    assert(len(f_ab_call.arguments) == 0)

    f_ab_call_xy = f_ab.call(core.Symbol("x"), core.Symbol("y"))
    assert(f_ab_call_xy.function_declaration == f_ab)
    assert(f_ab_call_xy.arguments[0] == core.Symbol("x"))
    assert(f_ab_call_xy.arguments[1] == core.Symbol("y"))

    Out = function.FunctionArgument.Out
    f_abc = function.FunctionDeclaration("f", [In("a"),In("b"),Out("c")])
    assert(f_abc.name == "f")
    assert(len(f_abc.arguments) == 3)
    assert(f_abc.list_dependees() == [Out("c")])
    assert(f_abc.list_dependencies() == [In("a"), In("b")])

    f_abc_call = f_abc.call()
    assert(f_abc_call.function_declaration == f_abc)
    assert(len(f_abc_call.arguments) == 0)

    f_abc_call_xyz = f_abc.call(core.Symbol("x"), core.Symbol("y"), core.Symbol("z"))
    assert(f_abc_call_xyz.function_declaration == f_abc)
    assert(f_abc_call_xyz.arguments[0] == core.Symbol("x"))
    assert(f_abc_call_xyz.arguments[1] == core.Symbol("y"))
    assert(f_abc_call_xyz.arguments[2] == core.Symbol("z"))

def test_FunctionCall():
    Symbol = core.Symbol
    Tuple = containers.Tuple
    In = function.FunctionArgument.In
    Out = function.FunctionArgument.Out
    f = function.FunctionDeclaration("f", [])
    f_ab = function.FunctionDeclaration("f", [In("a"),In("b"),])
    f_abc = function.FunctionDeclaration("f", [In("a"),In("b"),Out("c")])
    f_abcd = function.FunctionDeclaration("f", [In("a"),In("b"),Out("c"),Out("d")])
    
    f_ab_call = f_ab.call()
    f_ab_call_x = f_ab.call(Symbol("x"))
    f_ab_call_xy = f_ab.call(Symbol("x"), Symbol("y"))
    f_ab_call_ax = f_ab.call(a = Symbol("x"))
    f_ab_call_axby = f_ab.call(a = Symbol("x"), b = Symbol("y"))
    f_ab_call_xby = f_ab.call(Symbol("x"), b = Symbol("y"))

    f_abc_call = f_abc.call()
    f_abc_call_xy = f_abc.call(Symbol("x"), Symbol("y"))
    f_abc_call_xyz = f_abc.call(Symbol("x"), Symbol("y"), Symbol("z"))
    f_abc_call_xy_z = f_abc_call_xy.assign_to(Symbol("z"))
    
    f_abcd_call_xy = f_abc.call(Symbol("x"), Symbol("y"))
    f_abcd_call_xy_zw = f_abcd_call_xy.assign_to(Tuple(Symbol("z"), Symbol("w")))

    assert(f.call(Symbol("x")).arguments_valid() == False)
    assert(f.call(a=Symbol("x")).arguments_valid() == False)
    assert(f_ab_call_xy.arguments_valid() == True())
    
    assert(f_ab_call.unbound_arguments()
    assert(f_ab_call_x.arguments[0] == Symbol("x"))
    assert(f_ab_call_xy.arguments[0] == Symbol("x"))
    assert(f_ab_call_xy.arguments[1] == Symbol("y"))

    assert("a" in f_ab_call_ax.kwarguments)
    assert(f_ab_call_ax.kwarguments["a"] == Symbol("x"))
    
    assert(f_ab_call_xy.arguments[1] == Symbol("y"))
