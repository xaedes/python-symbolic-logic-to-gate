
import syloga.ast.core as core
import syloga.ast.containers as containers

def assert_copy_test(a):
    a_copy = a.copy()
    assert(a is not a_copy)
    assert(a == a_copy)

def test_Tuple():
    tpl = containers.Tuple()
    assert(tpl.args == tuple([]))
    assert(len(tpl) == 0)
    assert(tuple(iter(tpl)) == tpl.args)
    assert_copy_test(tpl)

    tpl = containers.Tuple(1,2,3)
    assert(tpl.args == (1,2,3))
    assert(len(tpl) == 3)
    assert(tuple(iter(tpl)) == tpl.args)
    assert(tpl[0] == 1)
    assert(tpl[1] == 2)
    assert(tpl[2] == 3)
    assert_copy_test(tpl)

def test_Dict():
    dct = containers.Dict()
    assert(dct.args == tuple([]))
    assert(len(dct) == 0)
    assert_copy_test(dct)

    dct = containers.Dict((0,"a"),(1,"b"),(2,"c"))
    assert(dct.args[0] == (0,"a"))
    assert(len(dct) == 3)
    assert(0 in dct)
    assert(1 in dct)
    assert(2 in dct)
    assert("a" not in dct)
    assert("b" not in dct)
    assert("c" not in dct)
    assert(dct[0] == "a")
    assert(dct[1] == "b")
    assert(dct[2] == "c")
    assert_copy_test(dct)
