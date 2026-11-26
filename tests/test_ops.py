"""The op list on CPU: every elementwise op is checked on some dev cases, and torch's call
takes out= on each of them, as the check calls it. On the card, isnan without an out= form
stopped the whole run at its first speed case."""

from sol_eval.ops import OPS


def test_every_elementwise_op_has_dev_cases():
    for name, op in OPS.items():
        assert op.dev_cases(), name
