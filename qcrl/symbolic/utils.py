import numpy as np

import sympy as sym
from sympy.physics.quantum import Operator, Dagger
from sympy import pprint, latex, powsimp

import warnings

# TO-DO, lambdify expressions before substitutions to speed up computation


def quantum_transformations(
    expr,
    only_op_terms=True,
    rwa_freq=None,
    rwa_args=None,
    time_symbol=None,
):
    """
    Applies Typical Approximations and Transformations that are used in QM.
    If only_op_terms = True, terms without operators in the expression are removed
    If a float value is passed for rwa_freq, terms rotating with frequencies of higher magnitude are neglected.
    To apply the Rotating Wave Approximation, it is assumed that the terms are already in their rotating frames.
    """
    if not isinstance(expr, sym.Basic):
        raise ValueError("the expression is not a sympy expression")

    for arg in expr.args:
        remove = False
        contains_op = False

        for node in sym.preorder_traversal(arg):
            if rwa_freq is not None and time_symbol is not None:
                if isinstance(node, sym.exp):
                    frequency_mag = sym.Abs(
                        node.args[0].subs(rwa_args) / sym.I / time_symbol
                    )
                    if frequency_mag >= rwa_freq:
                        remove = True
            elif rwa_freq is not None and time_symbol is None:
                if isinstance(node, sym.exp):
                    frequency_mag = sym.Abs(node.args[0].subs(rwa_args) / sym.I)
                    if frequency_mag > rwa_freq:
                        remove = True

            if isinstance(node, (Operator, Dagger)):
                contains_op = True

        if only_op_terms and not contains_op:
            remove = True

        if remove:
            expr -= arg

    return expr.copy()


sym.Basic.quantum_transformations = lambda self, only_op_terms, rwa_freq, rwa_args, time_symbol: quantum_transformations(
    self, only_op_terms, rwa_freq, rwa_args, time_symbol
)


def custom_printer(expr, name, subs_list=None, full_subs_list=None, cutoff_val=1e-3):
    if full_subs_list is not None:
        subs_expr = expr.expand()
        if not isinstance(subs_expr, sym.Add):
            raise TypeError(
                "the expression is not a sympy.Add method and can't be expanded to one, no terms can be neglected in this case"
            )
        temp_expr = sym.Abs(subs_expr.subs(full_subs_list))
        symbols_remaining = temp_expr.free_symbols
        if len(symbols_remaining) > 0:
            raise ValueError(
                f"Not enough symbol values are provided in the full_subs_list, including: {symbols_remaining}"
            )
        for arg in subs_expr.args:
            subs_arg = sym.Abs(arg.subs(full_subs_list))
            if subs_arg < cutoff_val:
                subs_expr -= arg
    print(f"###: {name}")
    pprint(expr, use_unicode=True)

    final_expr = expr

    if subs_list is not None:
        symbols_expr = powsimp(expr.subs(subs_list))
        final_expr = symbols_expr
        print("### substituted")
        pprint(symbols_expr)

    if full_subs_list is not None:
        final_expr = powsimp(subs_expr.subs(subs_list))
        print("### neglecting small values")
        pprint(final_expr)

    print(f"latex version of final expr: {latex(final_expr)}")
    print("###")
