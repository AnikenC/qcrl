import sympy as sym
from sympy.physics.quantum import Operator, Dagger
from sympy import pprint, latex, powsimp


def quantum_transformations(
    expr,
    only_op_terms=True,
    rwa_freq=None,
    rwa_args=None,
    time_symbol=None,
    evaluate=False,
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


sym.Basic.quantum_transformations = lambda self, only_op_terms, rwa_freq, rwa_args, time_symbol, evaluate: quantum_transformations(
    self, only_op_terms, rwa_freq, rwa_args, time_symbol, evaluate
)


def custom_printer(expr, name, subs_list=None):
    print(f"###: {name}")
    pprint(expr, use_unicode=True)
    if subs_list is not None:
        print("### substituted")
        expr = powsimp(expr.subs(subs_list))
        pprint(expr)
        print(f"latex version (with substitutions): {latex(expr)}")
    else:
        print(f"latex version: {latex(expr)}")
    print("###")
