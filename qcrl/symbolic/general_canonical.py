import sympy as sym
from sympy.core.operations import AssocOp
from sympy.physics.quantum import Commutator, Operator, Dagger
from typing import Tuple, List


def evaluate_with_ccr(expr, ccr_list: Tuple):
    """
    Evaluates an expression of Operators with the canonical commuation relations specified in the ccr_list
    expr: sympy expression
    ccr_list: tuple of sympy expressions, with each expression containing a commutator
    """

    if not isinstance(expr, sym.Basic):
        raise TypeError("The Expression to simplify is not a sympy expression")

    comm_list = []
    value_list = []
    for ccr in ccr_list:
        if not isinstance(ccr, sym.Eq):
            if isinstance(ccr, sym.Basic):
                ccr = sym.Eq(ccr, 0)
            else:
                raise TypeError(
                    "at least one of the canonical commutation relations is not a sympy expression"
                )

        for node in sym.preorder_traversal(ccr):
            if isinstance(node, Commutator):
                comm_list.append(node)
                break

    if len(comm_list) != len(ccr_list):
        raise ValueError(
            "One of the canonical commutation relations does not include a sympy commutator"
        )

    for i in range(len(ccr_list)):
        ccr = ccr_list[i]
        comm = comm_list[i]
        sol = sym.solve(ccr, comm)[0]
        value_list.append(sol)

    if len(value_list) != len(ccr_list):
        raise ValueError(
            "There are more than one solutions to at least one of the canonical commutation relations"
        )

    op_list = []
    op_list_1d = []

    for comm in comm_list:
        op_list.append([comm.args[0], comm.args[1]])
        op_list_1d.append(comm.args[0])
        op_list_1d.append(comm.args[1])

    list_of_ops = list(dict.fromkeys(op_list_1d))

    def is_power_of_op(expr):
        return (
            isinstance(expr, sym.Pow)
            and expr.args[0] in list_of_ops
            and isinstance(expr.args[1], sym.Number)
            and expr.args[1] >= 1
        )

    def in_op_list(op1: Operator, op2: Operator, list: List):
        return_list = []
        for sublist in list:
            if (op1 in sublist) and (op2 in sublist):
                return_list.append(sublist)
        return return_list

    for i in range(len(list_of_ops)):
        for j in range(len(list_of_ops)):
            if i != j and i < j:
                pair = in_op_list(list_of_ops[i], list_of_ops[j], op_list)
                if len(pair) != 1:
                    raise ValueError(
                        f"Operator {list_of_ops[i]} and {list_of_ops[j]} need a ccr defined"
                    )

    def get_value_from_ops(op1, op2, op_list, value_list):
        value = 0
        if op1 == op2:
            return op1, op2, value

        for i, sublist in enumerate(op_list):
            if op1 in sublist and op2 in sublist:
                a, b = sublist

                if a == op1 and b == op2:
                    return op1, op2, value

                elif b == op1 and a == op2:
                    value = value_list[i]
                    return op2, op1, value

                else:
                    raise ValueError("Something's wrong")

    def walk_tree(expr):
        if isinstance(expr, sym.Number):
            return expr

        if not isinstance(expr, AssocOp) and not isinstance(expr, sym.Function):
            return expr.copy()

        elif not isinstance(expr, sym.Mul):
            return expr.func(*(walk_tree(node) for node in expr.args))

        else:
            args = [arg for arg in expr.args]

            for i in range(len(args) - 1):
                x = args[i]
                y = args[i + 1]

                if isinstance(x, (Operator, Dagger)) and isinstance(
                    y, (Operator, Dagger)
                ):
                    op_a, op_b, val = get_value_from_ops(x, y, op_list, value_list)
                    if op_a != x:
                        args = args[0:i] + [op_a * op_b - val] + args[i + 2 :]
                        return walk_tree(sym.Mul(*args).expand())

                if is_power_of_op(x) and isinstance(y, (Operator, Dagger)):
                    temp_op = x.args[0]
                    op_a, op_b, val = get_value_from_ops(
                        temp_op, y, op_list, value_list
                    )
                    if op_a != temp_op:
                        args = (
                            args[0:i]
                            + [
                                op_a * sym.Pow(op_b, x.args[1])
                                - x.args[1] * sym.Pow(op_b, x.args[1] - 1) * val
                            ]
                            + args[i + 2 :]
                        )
                        return walk_tree(sym.Mul(*args).expand())

                if isinstance(x, (Operator, Dagger)) and is_power_of_op(y):
                    temp_op = y.args[0]
                    op_a, op_b, val = get_value_from_ops(
                        x, temp_op, op_list, value_list
                    )
                    if op_b != temp_op:
                        args = (
                            args[0:i]
                            + [
                                sym.Pow(op_a, y.args[1]) * op_b
                                - y.args[1] * sym.Pow(op_a, y.args[1] - 1) * val
                            ]
                            + args[i + 2 :]
                        )
                        return walk_tree(sym.Mul(*args).expand())

                if is_power_of_op(x) and is_power_of_op(y):
                    temp_x = x.args[0]
                    temp_y = y.args[0]
                    op_a, op_b, val = get_value_from_ops(
                        temp_x, temp_y, op_list, value_list
                    )
                    if op_a != temp_x:
                        args = (
                            args[0:i]
                            + [
                                sym.Pow(temp_x, x.args[1] - 1),
                                op_a * op_b - val,
                                sym.Pow(temp_y, y.args[1] - 1),
                            ]
                            + args[i + 2 :]
                        )
                        return walk_tree(sym.Mul(*args).expand())

            return expr.copy()

    return walk_tree(expr)


sym.Basic.evaluate_with_ccr = lambda self, ccr_list: evaluate_with_ccr(self, ccr_list)
