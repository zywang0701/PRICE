"""Readable reference implementation of PRICE (Algorithms 1 and 2 of the paper).

These files are written to be read. They mirror the equations of Appendix B one to one and
depend only on numpy. The numbers in the paper were produced by the experiment chain under
experiments/ (E17/allocation.py and E18/price_sweep.py for PRICE-oracle, E22/controller.py and
E22/models.py for PRICE-deployed); this package is the same logic without the replay machinery.
"""
from .vote import boltzmann_vote
from .oracle import QueryPrimitives, estimate_primitives, priced_choice, realized_budget, price_for_budget, frontier
from .deployed import predicted_gain, should_stop, blend_log_mgf, update_price, run_query
