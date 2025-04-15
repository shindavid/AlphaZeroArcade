from alphazero.servers.gaming.base_params import BaseParams
from alphazero.servers.gaming.server_base import ServerParams

from dataclasses import dataclass, fields
import logging


logger = logging.getLogger(__name__)


@dataclass
class BenchmarkServerParams(ServerParams):
    rating_tag: str = ''

    @staticmethod
    def add_additional_args(group):
        defaults = BenchmarkServerParams()
        group.add_argument('-r', '--rating-tag', default=defaults.rating_tag,
                           help='evaluation tag. Loop controller collates ratings by this str. It is '
                           'the responsibility of the user to make sure that the same '
                           'binary/params are used across different EvalServer processes '
                           'sharing the same rating-tag. (default: "%(default)s")')
