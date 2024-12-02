from dataclasses import dataclass
from decimal import Decimal


@dataclass
class PkTrailingStop:
    activation_delta: Decimal
    trailing_delta: Decimal
