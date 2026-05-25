"""Strong / new baseline factor miners.

Each subpackage exposes a `get_library() -> list[dict]` function returning a
factor library in the same shape as `tests.base_factors.get_combined_base_factor_library()`:

    [
        {
            "id":          str,
            "name":        str,
            "code":        str,   # Python class implementing AlphaFactorTemplate
            "description": str,
            "parameters":  dict,
            "category":    str,
            "evaluation":  dict,  # filled by run_baseline.py
        },
        ...
    ]
"""

from baselines.common import (
    FACTOR_TEMPLATE_HEADER,
    build_factor_record,
    call_evaluator_http,
    dump_library,
    load_library,
)

__all__ = [
    "FACTOR_TEMPLATE_HEADER",
    "build_factor_record",
    "call_evaluator_http",
    "dump_library",
    "load_library",
]
