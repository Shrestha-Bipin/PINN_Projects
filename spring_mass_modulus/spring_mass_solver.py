from sympy import Function, Number, Symbol

import modulus.sym
from modulus.sym.domain import Domain
from modulus.sym.domain.constraint import PointwiseBoundaryConstraint
from modulus.sym.eq.pde import PDE
from modulus.sym.geometry.primitives_1d import Point1D
from modulus.sym.hydra import ModulusConfig, instantiate_arch
from modulus.sym.key import Key
from modulus.sym.solver import Solver


class SpringMass(PDE):
    """Coupled three-mass spring ODE system."""

    name = "SpringMass"

    def __init__(self, k=(2, 1, 1, 2), m=(1, 1, 1)):
        self.k = k
        self.m = m

        k1, k2, k3, k4 = k
        m1, m2, m3 = m

        t = Symbol("t")
        input_variables = {"t": t}

        x1 = Function("x1")(*input_variables)
        x2 = Function("x2")(*input_variables)
        x3 = Function("x3")(*input_variables)

        k1 = self._to_symbolic(k1, input_variables)
        k2 = self._to_symbolic(k2, input_variables)
        k3 = self._to_symbolic(k3, input_variables)
        k4 = self._to_symbolic(k4, input_variables)

        m1 = self._to_symbolic(m1, input_variables)
        m2 = self._to_symbolic(m2, input_variables)
        m3 = self._to_symbolic(m3, input_variables)

        self.equations = {
            "ode_x1": m1 * x1.diff(t, 2) + (k1 + k2) * x1 - k2 * x2,
            "ode_x2": m2 * x2.diff(t, 2) + (k2 + k3) * x2 - k2 * x1 - k3 * x3,
            "ode_x3": m3 * x3.diff(t, 2) + (k3 + k4) * x3 - k3 * x2,
        }

    @staticmethod
    def _to_symbolic(value, input_variables):
        if isinstance(value, str):
            return Function(value)(*input_variables)
        if isinstance(value, (float, int)):
            return Number(value)
        return value


@modulus.sym.main(config_path="conf", config_name="config")
def run(cfg: ModulusConfig) -> None:
    spring_mass = SpringMass(k=(2, 1, 1, 2), m=(1, 1, 1))

    spring_mass_net = instantiate_arch(
        input_keys=[Key("t")],
        output_keys=[Key("x1"), Key("x2"), Key("x3")],
        cfg=cfg.arch.fully_connected,
    )

    nodes = spring_mass.make_nodes() + [
        spring_mass_net.make_node(name="spring_mass_network")
    ]

    geometry = Point1D(0)
    t_symbol = Symbol("t")
    t_max = 20.0

    domain = Domain()

    initial_condition = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geometry,
        outvar={
            "x1": 1.0,
            "x2": 0.0,
            "x3": 0.0,
            "x1__t": 0.0,
            "x2__t": 0.0,
            "x3__t": 0.0,
        },
        batch_size=cfg.batch_size.IC,
        parameterization={t_symbol: 0.0},
    )
    domain.add_constraint(initial_condition, name="IC")

    interior = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geometry,
        outvar={"ode_x1": 0.0, "ode_x2": 0.0, "ode_x3": 0.0},
        batch_size=cfg.batch_size.interior,
        parameterization={t_symbol: (0.0, t_max)},
    )
    domain.add_constraint(interior, name="interior")

    solver = Solver(cfg, domain)
    solver.solve()


if __name__ == "__main__":
    run()
