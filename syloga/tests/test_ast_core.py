
import syloga.ast.core as core

def assert_copy_test(a):
    a_copy = a.copy()
    assert(a is not a_copy)
    assert(a == a_copy)

def test_Symbol():
    a = core.Symbol("a")
    assert(a.func == core.Symbol)
    assert(a.name == "a")
    assert(str(a) == a.name)
    assert(repr(a) == a.name)
    
    a_0 = core.Symbol("a_0")
    assert(a_0.name == "a_0")
    assert(str(a_0) == a_0.name)
    assert(repr(a_0) == a_0.name)

    assert_copy_test(a)

def test_Indexed():
    a0 = core.Indexed("a", 0)
    assert(a0.func == core.Indexed)
    assert(a0.name == "a")
    assert(a0.index == 0)

    a01 = a0[1]
    assert(a01.name == a0)
    assert(a01.index == 1)

    assert_copy_test(a0)

def test_Indexable():
    a = core.Indexable("a")
    assert(a.func == core.Indexable)
    assert(a.name == "a")
    assert(a[0].name == a)
    assert(a[0].index == 0)

    assert(a[1].name == a)
    assert(a[1].index == 1)

    assert_copy_test(a)

def test_iter_Expression():
    e = core.Expression(core.Expression, (1,2,3))
    assert(e.func == core.Expression)
    assert(e.args == (1,2,3))
    assert(tuple(iter(e)) == e.args)

    assert_copy_test(e)

def test_BooleanExpression_UpToTwoArgs():
    a = core.Symbol("a")
    b = core.Symbol("b")

    not_a = ~a
    assert(type(not_a) == core.BooleanNot)
    assert(not_a.func == core.BooleanNot)
    assert(not_a.args[0] == a)
    assert_copy_test(not_a)

    a_and_b = a & b
    assert(type(a_and_b) == core.BooleanAnd)
    assert(a_and_b.func == core.BooleanAnd)
    assert(a_and_b.args[0] == a)
    assert(a_and_b.args[1] == b)
    assert_copy_test(a_and_b)

    a_or_b = a | b
    assert(type(a_or_b) == core.BooleanOr)
    assert(a_or_b.func == core.BooleanOr)
    assert(a_or_b.args[0] == a)
    assert(a_or_b.args[1] == b)
    assert_copy_test(a_or_b)

    a_xor_b = a ^ b
    assert(type(a_xor_b) == core.BooleanXor)
    assert(a_xor_b.func == core.BooleanXor)
    assert(a_xor_b.args[0] == a)
    assert(a_xor_b.args[1] == b)
    assert_copy_test(a_xor_b)

    not_a = core.BooleanNot(a)
    assert(type(not_a) == core.BooleanNot)
    assert(not_a.func == core.BooleanNot)
    assert(not_a.args[0] == a)

    a_and_b = core.BooleanAnd(a,b)
    assert(type(a_and_b) == core.BooleanAnd)
    assert(a_and_b.func == core.BooleanAnd)
    assert(a_and_b.args[0] == a)
    assert(a_and_b.args[1] == b)

    a_or_b = core.BooleanOr(a,b)
    assert(type(a_or_b) == core.BooleanOr)
    assert(a_or_b.func == core.BooleanOr)
    assert(a_or_b.args[0] == a)
    assert(a_or_b.args[1] == b)

    a_xor_b = core.BooleanXor(a,b)
    assert(type(a_xor_b) == core.BooleanXor)
    assert(a_xor_b.func == core.BooleanXor)
    assert(a_xor_b.args[0] == a)
    assert(a_xor_b.args[1] == b)

    a_nor_b = core.BooleanNor(a,b)
    assert(a_nor_b.func == core.BooleanNor)
    assert(a_nor_b.args[0] == a)
    assert(a_nor_b.args[1] == b)
    assert_copy_test(a_nor_b)

    a_nand_b = core.BooleanNand(a,b)
    assert(a_nand_b.func == core.BooleanNand)
    assert(a_nand_b.args[0] == a)
    assert(a_nand_b.args[1] == b)
    assert_copy_test(a_nand_b)

def test_BooleanExpression_ThreeArgs():
    a = core.Symbol("a")
    b = core.Symbol("b")
    c = core.Symbol("c")

    and_abc = core.BooleanAnd(a,b,c)
    assert(type(and_abc) == core.BooleanAnd)
    assert(and_abc.func == core.BooleanAnd)
    assert(and_abc.args[0] == a)
    assert(and_abc.args[1] == b)
    assert(and_abc.args[2] == c)
    assert_copy_test(and_abc)

    or_abc = core.BooleanOr(a,b,c)
    assert(type(or_abc) == core.BooleanOr)
    assert(or_abc.func == core.BooleanOr)
    assert(or_abc.args[0] == a)
    assert(or_abc.args[1] == b)
    assert(or_abc.args[2] == c)
    assert_copy_test(or_abc)

    xor_abc = core.BooleanXor(a,b,c)
    assert(type(xor_abc) == core.BooleanXor)
    assert(xor_abc.func == core.BooleanXor)
    assert(xor_abc.args[0] == a)
    assert(xor_abc.args[1] == b)
    assert(xor_abc.args[2] == c)
    assert_copy_test(xor_abc)

    nor_abc = core.BooleanNor(a,b,c)
    assert(type(nor_abc) == core.BooleanNor)
    assert(nor_abc.func == core.BooleanNor)
    assert(nor_abc.args[0] == a)
    assert(nor_abc.args[1] == b)
    assert(nor_abc.args[2] == c)
    assert_copy_test(nor_abc)

    nand_abc = core.BooleanNand(a,b,c)
    assert(type(nand_abc) == core.BooleanNand)
    assert(nand_abc.func == core.BooleanNand)
    assert(nand_abc.args[0] == a)
    assert(nand_abc.args[1] == b)
    assert(nand_abc.args[2] == c)
    assert_copy_test(nand_abc)

