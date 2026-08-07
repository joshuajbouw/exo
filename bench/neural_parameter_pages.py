"""Executable algebra for the neural-parameter-page experiment.

This is deliberately a small simulator, not an inference implementation.  It
mechanically checks the contracts in ``NEURAL_PARAMETER_PAGES.md`` before the
same seams are introduced into an MLX model.
"""

from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import dataclass

type Vector = tuple[float, ...]
type Matrix = tuple[Vector, ...]

_IDENTITY_CONTEXT = b"exo.neural-parameter-pages.simulator.v1\0"


class InvalidPageError(ValueError):
    """The page does not satisfy the registered ABI."""


class AmbiguousActivationError(ValueError):
    """More than one page claims a site without a composition contract."""


class PerturbationBudgetExceededError(ValueError):
    """Selected page norms exceed the site's registered budget."""


def _field(value: bytes) -> bytes:
    return struct.pack(">Q", len(value)) + value


def _text(value: str) -> bytes:
    return _field(value.encode("utf-8"))


def _identity(*parts: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(_IDENTITY_CONTEXT)
    for part in parts:
        digest.update(_field(part))
    return digest.hexdigest()


def _matrix_bytes(matrix: Matrix) -> bytes:
    encoded = bytearray(struct.pack(">I", len(matrix)))
    for row in matrix:
        encoded.extend(struct.pack(">I", len(row)))
        for value in row:
            if not math.isfinite(value):
                raise InvalidPageError("page matrices must contain finite values")
            encoded.extend(struct.pack(">d", value))
    return bytes(encoded)


def _matrix_vector(matrix: Matrix, vector: Vector) -> Vector:
    if any(len(row) != len(vector) for row in matrix):
        raise InvalidPageError("matrix and vector dimensions do not agree")
    return tuple(
        math.fsum(value * item for value, item in zip(row, vector, strict=True))
        for row in matrix
    )


def _frobenius_norm(matrix: Matrix) -> float:
    return math.sqrt(math.fsum(value * value for row in matrix for value in row))


@dataclass(frozen=True, slots=True)
class SitePage:
    """One low-rank residual at one named projection site."""

    site_name: str
    input_width: int
    output_width: int
    rank: int
    scale: float
    matrix_a: Matrix
    matrix_b: Matrix

    def __post_init__(self) -> None:
        if not self.site_name:
            raise InvalidPageError("site name must not be empty")
        if self.input_width < 1 or self.output_width < 1 or self.rank < 1:
            raise InvalidPageError("page dimensions and rank must be positive")
        if not math.isfinite(self.scale):
            raise InvalidPageError("page scale must be finite")
        if len(self.matrix_a) != self.rank:
            raise InvalidPageError("matrix A must have rank rows")
        if any(len(row) != self.input_width for row in self.matrix_a):
            raise InvalidPageError("matrix A input width is inconsistent")
        if len(self.matrix_b) != self.output_width:
            raise InvalidPageError("matrix B must have output-width rows")
        if any(len(row) != self.rank for row in self.matrix_b):
            raise InvalidPageError("matrix B rank is inconsistent")
        _matrix_bytes(self.matrix_a)
        _matrix_bytes(self.matrix_b)

    @property
    def parameter_count(self) -> int:
        return self.rank * (self.input_width + self.output_width)

    @property
    def perturbation_bound(self) -> float:
        return (
            abs(self.scale / self.rank)
            * _frobenius_norm(self.matrix_b)
            * _frobenius_norm(self.matrix_a)
        )

    def residual(self, vector: Vector) -> Vector:
        if len(vector) != self.input_width:
            raise InvalidPageError("page input does not match its registered width")
        compressed = _matrix_vector(self.matrix_a, vector)
        expanded = _matrix_vector(self.matrix_b, compressed)
        factor = self.scale / self.rank
        return tuple(factor * value for value in expanded)

    def canonical_bytes(self) -> bytes:
        return b"".join(
            (
                _text(self.site_name),
                struct.pack(">III", self.input_width, self.output_width, self.rank),
                struct.pack(">d", self.scale),
                _field(_matrix_bytes(self.matrix_a)),
                _field(_matrix_bytes(self.matrix_b)),
            )
        )


@dataclass(frozen=True, slots=True)
class ParameterPage:
    """A content-identified, privacy-domain-scoped neural extension."""

    base_model_id: str
    runtime_profile_id: str
    training_recipe_id: str
    evidence_closure_id: str
    privacy_domain_id: str
    page_version: int
    sites: tuple[SitePage, ...]

    def __post_init__(self) -> None:
        identifiers = (
            self.base_model_id,
            self.runtime_profile_id,
            self.training_recipe_id,
            self.evidence_closure_id,
            self.privacy_domain_id,
        )
        if any(not identifier for identifier in identifiers):
            raise InvalidPageError("page identities and evidence must not be empty")
        if self.page_version < 1:
            raise InvalidPageError("page version must be positive")
        if not self.sites:
            raise InvalidPageError("a page must contain at least one site")
        site_names = tuple(site.site_name for site in self.sites)
        if len(set(site_names)) != len(site_names):
            raise InvalidPageError("a page may name each site only once")
        if site_names != tuple(sorted(site_names)):
            raise InvalidPageError("page sites must be in canonical name order")

    @property
    def page_id(self) -> str:
        return _identity(b"page", self.canonical_bytes())

    def canonical_bytes(self) -> bytes:
        return b"".join(
            (
                _text(self.base_model_id),
                _text(self.runtime_profile_id),
                _text(self.training_recipe_id),
                _text(self.evidence_closure_id),
                _text(self.privacy_domain_id),
                struct.pack(">Q", self.page_version),
                struct.pack(">I", len(self.sites)),
                *(_field(site.canonical_bytes()) for site in self.sites),
            )
        )


@dataclass(frozen=True, slots=True, order=True)
class DomainMembership:
    principal_id: str
    privacy_domain_id: str


@dataclass(frozen=True, slots=True, order=True)
class PageGrant:
    principal_id: str
    page_id: str


@dataclass(frozen=True, slots=True, order=True)
class PageService:
    page_id: str
    concept: str
    first_epoch: int
    last_epoch: int

    def __post_init__(self) -> None:
        if self.first_epoch < 0 or self.last_epoch < self.first_epoch:
            raise ValueError("service epoch interval is invalid")


@dataclass(frozen=True, slots=True)
class ActivationRequest:
    invocation_id: str
    principal_id: str
    privacy_domain_id: str
    concept: str
    epoch: int
    base_model_id: str
    runtime_profile_id: str
    site_budgets: tuple[tuple[str, float], ...]

    def budget_for(self, site_name: str) -> float:
        matches = tuple(value for name, value in self.site_budgets if name == site_name)
        if len(matches) != 1 or not math.isfinite(matches[0]) or matches[0] < 0.0:
            raise ValueError(f"missing or invalid budget for site {site_name!r}")
        return matches[0]

    def canonical_bytes(self) -> bytes:
        budgets = tuple(sorted(self.site_budgets))
        return b"".join(
            (
                _text(self.invocation_id),
                _text(self.principal_id),
                _text(self.privacy_domain_id),
                _text(self.concept),
                struct.pack(">Q", self.epoch),
                _text(self.base_model_id),
                _text(self.runtime_profile_id),
                struct.pack(">I", len(budgets)),
                *(_text(name) + struct.pack(">d", value) for name, value in budgets),
            )
        )


@dataclass(frozen=True, slots=True)
class ActivationProof:
    request: ActivationRequest
    fact_snapshot_id: str
    rule_profile_id: str
    selected_page_ids: tuple[str, ...]

    @property
    def proof_id(self) -> str:
        return _identity(
            b"activation-proof",
            self.request.canonical_bytes(),
            _text(self.fact_snapshot_id),
            _text(self.rule_profile_id),
            *(bytes.fromhex(page_id) for page_id in self.selected_page_ids),
        )


class TensorLogicPageSelector:
    """A T=0 relation join that proposes only granted, compatible pages."""

    def __init__(
        self,
        pages: tuple[ParameterPage, ...],
        memberships: tuple[DomainMembership, ...],
        grants: tuple[PageGrant, ...],
        services: tuple[PageService, ...],
        *,
        rule_profile_id: str,
    ) -> None:
        if not rule_profile_id:
            raise ValueError("rule profile id must not be empty")
        self._pages = {page.page_id: page for page in pages}
        if len(self._pages) != len(pages):
            raise ValueError("duplicate page identity")
        self._memberships = tuple(sorted(memberships))
        self._grants = tuple(sorted(grants))
        self._services = tuple(sorted(services))
        self._rule_profile_id = rule_profile_id

    @property
    def fact_snapshot_id(self) -> str:
        facts: list[bytes] = []
        facts.extend(
            b"membership" + _text(fact.principal_id) + _text(fact.privacy_domain_id)
            for fact in self._memberships
        )
        facts.extend(
            b"grant" + _text(fact.principal_id) + _text(fact.page_id)
            for fact in self._grants
        )
        facts.extend(
            b"service"
            + _text(fact.page_id)
            + _text(fact.concept)
            + struct.pack(">QQ", fact.first_epoch, fact.last_epoch)
            for fact in self._services
        )
        return _identity(b"fact-snapshot", *facts)

    def select(self, request: ActivationRequest) -> ActivationProof:
        membership = DomainMembership(request.principal_id, request.privacy_domain_id)
        if membership not in self._memberships:
            return self._proof(request, ())

        grants = {
            fact.page_id
            for fact in self._grants
            if fact.principal_id == request.principal_id
        }
        candidates: list[ParameterPage] = []
        for service in self._services:
            page = self._pages.get(service.page_id)
            if page is None or service.page_id not in grants:
                continue
            if service.concept != request.concept:
                continue
            if not service.first_epoch <= request.epoch <= service.last_epoch:
                continue
            if page.privacy_domain_id != request.privacy_domain_id:
                continue
            if page.base_model_id != request.base_model_id:
                continue
            if page.runtime_profile_id != request.runtime_profile_id:
                continue
            candidates.append(page)

        sites: dict[str, list[SitePage]] = {}
        for page in candidates:
            for site in page.sites:
                sites.setdefault(site.site_name, []).append(site)
        conflicts = tuple(name for name, values in sites.items() if len(values) > 1)
        if conflicts:
            raise AmbiguousActivationError(
                f"multiple pages claim sites without a composition contract: {sorted(conflicts)}"
            )
        for site_name, values in sites.items():
            if values[0].perturbation_bound > request.budget_for(site_name):
                raise PerturbationBudgetExceededError(site_name)
        return self._proof(request, tuple(sorted(page.page_id for page in candidates)))

    def _proof(
        self, request: ActivationRequest, selected_page_ids: tuple[str, ...]
    ) -> ActivationProof:
        return ActivationProof(
            request=request,
            fact_snapshot_id=self.fact_snapshot_id,
            rule_profile_id=self._rule_profile_id,
            selected_page_ids=selected_page_ids,
        )


def apply_projection(
    weight: Matrix,
    vector: Vector,
    *,
    site_name: str,
    proof: ActivationProof,
    pages: tuple[ParameterPage, ...],
) -> Vector:
    """Apply one base projection and only the pages named by the proof."""

    base = _matrix_vector(weight, vector)
    if not proof.selected_page_ids:
        return base

    by_id = {page.page_id: page for page in pages}
    result = base
    for page_id in proof.selected_page_ids:
        page = by_id.get(page_id)
        if page is None:
            raise InvalidPageError("activation proof names an unavailable page")
        matching = tuple(site for site in page.sites if site.site_name == site_name)
        if not matching:
            continue
        residual = matching[0].residual(vector)
        result = tuple(
            value + delta for value, delta in zip(result, residual, strict=True)
        )
    return result
