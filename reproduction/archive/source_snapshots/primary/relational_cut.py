"""Exact interaction cuts for a fixed two-predicate incidence-bag join.

The compressed source/left/key/right/sink flow is a classical biclique
compression construction. This module connects its checked minimum cut to
an exact affine residual and truthful retained-only output materialization.
It does not calibrate count evidence, infer truth, or claim an end-to-end
cost advantage. All arithmetic affecting a cut certificate is integral.

An input incidence is one bag occurrence of a row at a key. Multiple matching
keys retain multiple output derivations. This is not OR-key pair deduplication.
The oracle identity sets on the two sides must be disjoint; namespace distinct
predicates explicitly. Cross-side self-aliases require separate preprocessing.
"""
from __future__ import annotations

from collections import deque
from collections.abc import Hashable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from fractions import Fraction
from math import lcm
from numbers import Integral, Rational
from typing import Any


class ContractError(ValueError):
    """An input violates the fixed incidence-bag/oracle contract."""


class CertificateError(ValueError):
    """A supplied cut/flow witness fails exact independent verification."""


class NonlinearResidualError(ValueError):
    """Some key still contains unknown identities on both sides."""


class IncompleteExecutionError(ValueError):
    """A relevant retained identity lacks a truthful supplied label."""


@dataclass(frozen=True)
class Incidence:
    row_id: Hashable
    identity: Hashable
    key: Hashable
    multiplicity: int = 1
    payload: Any = None


@dataclass(frozen=True)
class Arc:
    tail: int
    head: int
    capacity: int
    structural: bool = False


@dataclass(frozen=True)
class FlowCertificate:
    """Exact integral max-flow/min-cut witnesses for a reconstructed network."""
    scale: int
    nodes: tuple[tuple[str, Any], ...]
    arcs: tuple[Arc, ...]
    flows: tuple[int, ...]
    source_side: frozenset[int]
    value: Fraction


@dataclass(frozen=True)
class CutPlan:
    target_version: str
    oracle_version: str
    cache_facts: tuple[tuple[Hashable, int], ...]
    acquire: tuple[Hashable, ...]
    cost: Fraction
    certificate: FlowCertificate


@dataclass(frozen=True)
class CutVerification:
    cost: Fraction
    node_count: int
    arc_count: int
    structural_arcs: int
    capacity_scale: int
    zero_primal_dual_gap: bool = True


@dataclass(frozen=True)
class AffineResidual:
    target_version: str
    oracle_version: str
    cache_facts: tuple[tuple[Hashable, int], ...]
    constant: int
    weights: tuple[tuple[Hashable, int], ...]

    def weight_map(self) -> dict[Hashable, int]:
        """Return a copy; zero-weight unknown identities remain explicit."""
        return dict(self.weights)


@dataclass(frozen=True)
class JoinedRow:
    left_row_id: Hashable
    right_row_id: Hashable
    key: Hashable
    multiplicity: int
    left_payload: Any = None
    right_payload: Any = None


def _hashable(value: Any, name: str) -> None:
    try:
        hash(value)
    except (TypeError, ValueError) as exc:
        raise ContractError(f'{name} must be hashable') from exc


def _integer(value: Any, name: str, *, nonnegative: bool = True) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ContractError(f'{name} must be an integer, not bool/float')
    result = int(value)
    if nonnegative and result < 0:
        raise ContractError(f'{name} must be nonnegative')
    return result


def _price(value: Any) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, Rational):
        raise ContractError('Prices must be exact Rational values (e.g. int or Fraction), not floats')
    result = Fraction(int(value.numerator), int(value.denominator))
    if result < 0:
        raise ContractError('Prices must be nonnegative')
    return result


def _max_flow(nodes: int, arcs: tuple[Arc, ...]) -> tuple[tuple[int, ...], frozenset[int]]:
    """Iterative Dinic with exact integers; no recursion-depth dependence.

    This portable reference solver makes no near-linear or large-scale claim.
    Every augmentation saturates a level-graph arc. The verifier does not trust
    this routine's traversal, residual graph, or termination condition.
    """
    adjacency: list[list[int]] = [[] for _ in range(nodes)]
    heads: list[int] = []
    residual: list[int] = []
    for arc in arcs:
        forward = len(heads)
        adjacency[arc.tail].append(forward)
        adjacency[arc.head].append(forward + 1)
        heads.extend((arc.head, arc.tail))
        residual.extend((arc.capacity, 0))
    source, sink = 0, 1
    while True:
        levels = [-1] * nodes
        levels[source] = 0
        queue = deque([source])
        while queue:
            u = queue.popleft()
            for edge in adjacency[u]:
                v = heads[edge]
                if residual[edge] > 0 and levels[v] < 0:
                    levels[v] = levels[u] + 1
                    queue.append(v)
        if levels[sink] < 0:
            partition = frozenset(i for i, level in enumerate(levels) if level >= 0)
            return tuple(arc.capacity - residual[2 * i] for i, arc in enumerate(arcs)), partition
        position = [0] * nodes
        while True:
            path_nodes = [source]
            path_edges: list[int] = []
            while path_nodes and path_nodes[-1] != sink:
                u = path_nodes[-1]
                while position[u] < len(adjacency[u]):
                    edge = adjacency[u][position[u]]
                    if residual[edge] > 0 and levels[heads[edge]] == levels[u] + 1:
                        path_edges.append(edge)
                        path_nodes.append(heads[edge])
                        break
                    position[u] += 1
                else:
                    path_nodes.pop()
                    if path_edges:
                        path_edges.pop()
                    if path_nodes:
                        position[path_nodes[-1]] += 1
            if not path_nodes:
                break
            amount = min(residual[edge] for edge in path_edges)
            for edge in path_edges:
                residual[edge] -= amount
                residual[edge ^ 1] += amount


class FactorizedBagJoin:
    """Fixed bag join whose uncertain leaves have disjoint left/right namespaces.

    Query structure and exact prices are copied at construction. Hashable keys
    and ids follow Python mapping equality semantics; callers must keep their
    hash/equality behavior stable. Cache labels are caller-supplied truthful
    immutable oracle facts, never random audit observations by implication.
    """

    def __init__(self, left: Iterable[Incidence], right: Iterable[Incidence],
                 prices: Mapping[Hashable, Rational], *, target_version: str,
                 oracle_version: str):
        if not isinstance(target_version, str) or not target_version:
            raise ContractError('A nonempty target_version is required')
        if not isinstance(oracle_version, str) or not oracle_version:
            raise ContractError('A nonempty oracle_version is required')
        if not isinstance(prices, Mapping):
            raise ContractError('prices must be a mapping')
        self.target_version, self.oracle_version = target_version, oracle_version
        self._left, left_order = self._read_side(left, 'left')
        self._right, right_order = self._read_side(right, 'right')
        if set(left_order) & set(right_order):
            raise ContractError('The same oracle identity occurs on both sides; namespace predicates or preprocess aliases')
        self._identities = tuple(left_order) + tuple(right_order)
        identity_set = set(self._identities)
        if set(prices) != identity_set:
            raise ContractError('Prices must name exactly every unique input oracle identity, including zero incidences')
        self._prices = {identity: _price(prices[identity]) for identity in self._identities}
        self._left_ids, self._right_ids = frozenset(left_order), frozenset(right_order)
        self._by_key: dict[Hashable, tuple[dict[Hashable, int], dict[Hashable, int]]] = {}
        self._left_rows: dict[Hashable, list[Incidence]] = {}
        self._right_rows: dict[Hashable, list[Incidence]] = {}
        for side, incidences, row_groups in ((0, self._left, self._left_rows),
                                              (1, self._right, self._right_rows)):
            for incidence in incidences:
                if not incidence.multiplicity:
                    continue
                self._by_key.setdefault(incidence.key, ({}, {}))[side].setdefault(incidence.identity, 0)
                self._by_key[incidence.key][side][incidence.identity] += incidence.multiplicity
                row_groups.setdefault(incidence.key, []).append(incidence)

    @staticmethod
    def _read_side(items: Iterable[Incidence], side: str) -> tuple[tuple[Incidence, ...], dict[Hashable, None]]:
        rows: dict[Hashable, Hashable] = {}
        identities: dict[Hashable, None] = {}
        result = []
        try:
            iterator = iter(items)
        except TypeError as exc:
            raise ContractError(f'{side} incidences must be iterable') from exc
        for item in iterator:
            if not isinstance(item, Incidence):
                raise ContractError(f'{side} items must be Incidence objects')
            _hashable(item.row_id, 'row_id')
            _hashable(item.identity, 'oracle identity')
            _hashable(item.key, 'key')
            multiplicity = _integer(item.multiplicity, 'incidence multiplicity')
            # Mapping-key equality includes identity for non-reflexive hashable
            # objects (for example, reusing the same float('nan') object).
            if item.row_id in rows and rows[item.row_id] not in {item.identity}:
                raise ContractError(f'One {side} row_id maps to conflicting oracle identities')
            rows[item.row_id] = item.identity
            identities.setdefault(item.identity, None)
            result.append(Incidence(item.row_id, item.identity, item.key, multiplicity, item.payload))
        return tuple(result), identities

    @property
    def identities(self) -> tuple[Hashable, ...]:
        return self._identities

    @property
    def incidences(self) -> tuple[tuple[Incidence, ...], tuple[Incidence, ...]]:
        return self._left, self._right

    def _facts(self, cache: Mapping[Hashable, int] | None) -> dict[Hashable, int]:
        if cache is None:
            return {}
        if not isinstance(cache, Mapping):
            raise ContractError('Cache/labels must be a mapping')
        result = {}
        for identity, value in cache.items():
            _hashable(identity, 'cache identity')
            if identity not in self._prices:
                raise ContractError('Cache contains an out-of-target oracle identity')
            if not isinstance(value, Integral) or int(value) not in (0, 1):
                raise ContractError('Cache values must be binary integer/Boolean labels')
            result[identity] = int(value)
        return result

    def _snapshot(self, facts: Mapping[Hashable, int]) -> tuple[tuple[Hashable, int], ...]:
        return tuple((identity, facts[identity]) for identity in self._identities if identity in facts)

    def _network(self, facts: Mapping[Hashable, int]) -> tuple[tuple[tuple[str, Any], ...], tuple[Arc, ...], int]:
        active_keys = []
        active_left: dict[Hashable, None] = {}
        active_right: dict[Hashable, None] = {}
        for key, (left, right) in self._by_key.items():
            unknown_left = tuple(identity for identity in left if identity not in facts)
            unknown_right = tuple(identity for identity in right if identity not in facts)
            if unknown_left and unknown_right:
                active_keys.append((key, unknown_left, unknown_right))
                active_left.update(dict.fromkeys(unknown_left))
                active_right.update(dict.fromkeys(unknown_right))
        scale = 1
        for identity in (*active_left, *active_right):
            scale = lcm(scale, self._prices[identity].denominator)
        total_price = sum((self._prices[v] for v in (*active_left, *active_right)), Fraction(0))
        infinity = int((1 + total_price) * scale)
        nodes: list[tuple[str, Any]] = [('source', None), ('sink', None)]
        left_index, right_index, key_index = {}, {}, {}
        for name, identities, index in (('left', active_left, left_index), ('right', active_right, right_index)):
            for identity in identities:
                index[identity] = len(nodes)
                nodes.append((name, identity))
        for key, _, _ in active_keys:
            key_index[key] = len(nodes)
            nodes.append(('key', key))
        arcs = [Arc(0, index, int(self._prices[identity] * scale)) for identity, index in left_index.items()]
        for key, left, right in active_keys:
            middle = key_index[key]
            arcs.extend(Arc(left_index[identity], middle, infinity, True) for identity in left)
            arcs.extend(Arc(middle, right_index[identity], infinity, True) for identity in right)
        arcs.extend(Arc(index, 1, int(self._prices[identity] * scale)) for identity, index in right_index.items())
        return tuple(nodes), tuple(arcs), scale

    @staticmethod
    def _partition_cover(nodes: tuple[tuple[str, Any], ...], source_side: frozenset[int]) -> tuple[Hashable, ...]:
        return tuple(identity for i, (kind, identity) in enumerate(nodes)
                     if (kind == 'left' and i not in source_side) or (kind == 'right' and i in source_side))

    def plan_cut(self, cache: Mapping[Hashable, int] | None = None) -> CutPlan:
        """Return a minimum-price fixed structural acquisition set with a witness."""
        facts = self._facts(cache)
        nodes, arcs, scale = self._network(facts)
        flows, partition = _max_flow(len(nodes), arcs)
        value = Fraction(sum(flow for arc, flow in zip(arcs, flows) if arc.tail == 0), scale)
        cover = self._partition_cover(nodes, partition)
        certificate = FlowCertificate(scale, nodes, arcs, flows, partition, value)
        plan = CutPlan(self.target_version, self.oracle_version, self._snapshot(facts), cover, value, certificate)
        self.verify_cut(plan, facts)
        return plan

    def verify_cut(self, plan: CutPlan, cache: Mapping[Hashable, int] | None = None) -> CutVerification:
        """Rebuild and verify capacities, primal flow, cut and zero duality gap."""
        facts = self._facts(cache)
        if not isinstance(plan, CutPlan) or not isinstance(plan.certificate, FlowCertificate):
            raise CertificateError('Expected a CutPlan and FlowCertificate')
        if (plan.target_version, plan.oracle_version) != (self.target_version, self.oracle_version):
            raise CertificateError('Cut version mismatch')
        if plan.cache_facts != self._snapshot(facts):
            raise CertificateError('Cut cache snapshot mismatch')
        nodes, arcs, scale = self._network(facts)
        cert = plan.certificate
        if isinstance(cert.scale, bool) or not isinstance(cert.scale, Integral) or cert.scale < 1:
            raise CertificateError('Invalid capacity scale')
        if not isinstance(cert.arcs, tuple) or any(
                not isinstance(arc, Arc)
                or any(isinstance(v, bool) or not isinstance(v, Integral) for v in (arc.tail, arc.head, arc.capacity))
                or not isinstance(arc.structural, bool) for arc in cert.arcs):
            raise CertificateError('Malformed integral certificate arc')
        if cert.nodes != nodes or cert.arcs != arcs or cert.scale != scale:
            raise CertificateError('Certificate network/capacity/scale mismatch')
        if not isinstance(cert.flows, tuple) or len(cert.flows) != len(arcs):
            raise CertificateError('Wrong number of arc flows')
        if not isinstance(cert.source_side, frozenset) or 0 not in cert.source_side or 1 in cert.source_side:
            raise CertificateError('Invalid source/sink cut partition')
        if any(isinstance(v, bool) or not isinstance(v, Integral) or not 0 <= v < len(nodes) for v in cert.source_side):
            raise CertificateError('Cut contains an invalid node index')
        balances = [0] * len(nodes)
        for arc, flow in zip(arcs, cert.flows):
            if isinstance(flow, bool) or not isinstance(flow, Integral) or not 0 <= flow <= arc.capacity:
                raise CertificateError('Flow violates integral capacity bounds')
            balances[arc.tail] -= int(flow)
            balances[arc.head] += int(flow)
        value = -balances[0]
        if balances[1] != value or any(balances[v] != 0 for v in range(2, len(nodes))):
            raise CertificateError('Flow conservation fails')
        cut_capacity = 0
        for arc in arcs:
            if arc.tail in cert.source_side and arc.head not in cert.source_side:
                if arc.structural:
                    raise CertificateError('A structural/infinite arc crosses the cut')
                cut_capacity += arc.capacity
        if value != cut_capacity:
            raise CertificateError('Nonzero primal/dual gap')
        cover = self._partition_cover(nodes, cert.source_side)
        if not isinstance(plan.acquire, tuple) or plan.acquire != cover:
            raise CertificateError('Returned acquisition set does not match the cut')
        price = sum((self._prices[v] for v in cover), Fraction(0))
        if not isinstance(plan.cost, Fraction) or not isinstance(cert.value, Fraction):
            raise CertificateError('Exact Fraction cut values are required')
        if price != plan.cost or cert.value != price or price != Fraction(value, scale):
            raise CertificateError('Cut objective/price mismatch')
        return CutVerification(price, len(nodes), len(arcs), sum(arc.structural for arc in arcs), scale)

    def linearize(self, cache: Mapping[Hashable, int]) -> AffineResidual:
        """Compute exact K/weights by key aggregates; fail if any interaction remains."""
        facts = self._facts(cache)
        weights = {identity: 0 for identity in self._identities if identity not in facts}
        constant = 0
        for left, right in self._by_key.values():
            unknown_left = [(v, m) for v, m in left.items() if v not in facts]
            unknown_right = [(v, m) for v, m in right.items() if v not in facts]
            if unknown_left and unknown_right:
                raise NonlinearResidualError('An unresolved key still contains unknown identities on both sides')
            positive_left = sum(m for v, m in left.items() if facts.get(v) == 1)
            positive_right = sum(m for v, m in right.items() if facts.get(v) == 1)
            constant += positive_left * positive_right
            for identity, multiplicity in unknown_left:
                weights[identity] += multiplicity * positive_right
            for identity, multiplicity in unknown_right:
                weights[identity] += multiplicity * positive_left
        return AffineResidual(self.target_version, self.oracle_version, self._snapshot(facts), constant, tuple(weights.items()))

    def _execution(self, labels: Mapping[Hashable, int], discard: Iterable[Hashable]) -> dict[Hashable, int]:
        facts = self._facts(labels)
        try:
            discarded = set(discard)
        except (TypeError, ValueError) as exc:
            raise ContractError('Discard identities must be a hashable iterable') from exc
        if not discarded <= self._prices.keys():
            raise ContractError('Discard contains an out-of-target oracle identity')
        if discarded & facts.keys():
            raise ContractError('Discard must contain only unknown identities; known labels are preserved')
        for left, right in self._by_key.values():
            live_left = [v for v in left if v not in discarded and facts.get(v) != 0]
            live_right = [v for v in right if v not in discarded and facts.get(v) != 0]
            if live_left and live_right and any(v not in facts for v in (*live_left, *live_right)):
                raise IncompleteExecutionError('A relevant retained identity is unknown and not explicitly discarded')
        return facts

    def count_output(self, labels: Mapping[Hashable, int], *, discard: Iterable[Hashable] = ()) -> int:
        """Count the complete retained bag without enumerating joined pairs."""
        facts = self._execution(labels, discard)
        return sum(sum(m for v, m in left.items() if facts.get(v) == 1)
                   * sum(m for v, m in right.items() if facts.get(v) == 1)
                   for left, right in self._by_key.values())

    def iter_joined_rows(self, labels: Mapping[Hashable, int], *, discard: Iterable[Hashable] = ()) -> Iterator[JoinedRow]:
        """Stream positive incidence pairs with integer bag multiplicities.

        Validation occurs on this call. Iterating materializes only output
        pairs; unlike graph construction, producing a flat result necessarily
        pays for its size. Rows repeated at distinct keys remain repeated.
        """
        facts = self._execution(labels, discard)
        def generate() -> Iterator[JoinedRow]:
            for key, left_rows in self._left_rows.items():
                positive_right = [row for row in self._right_rows.get(key, ()) if facts.get(row.identity) == 1]
                if not positive_right:
                    continue
                for left in left_rows:
                    if facts.get(left.identity) != 1:
                        continue
                    for right in positive_right:
                        yield JoinedRow(left.row_id, right.row_id, key, left.multiplicity * right.multiplicity,
                                        left.payload, right.payload)
        return generate()

    def iter_occurrences(self, labels: Mapping[Hashable, int], *, discard: Iterable[Hashable] = ()) -> Iterator[JoinedRow]:
        """Lazily expand the retained bag into rows of multiplicity one."""
        rows = self.iter_joined_rows(labels, discard=discard)
        def generate() -> Iterator[JoinedRow]:
            for row in rows:
                unit = JoinedRow(row.left_row_id, row.right_row_id, row.key, 1, row.left_payload, row.right_payload)
                for _ in range(row.multiplicity):
                    yield unit
        return generate()
