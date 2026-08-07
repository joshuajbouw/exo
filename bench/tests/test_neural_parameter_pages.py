# type: ignore

from __future__ import annotations

import struct

import pytest

from bench.neural_parameter_pages import (
    ActivationRequest,
    AmbiguousActivationError,
    DomainMembership,
    PageGrant,
    PageService,
    ParameterPage,
    PerturbationBudgetExceededError,
    SitePage,
    TensorLogicPageSelector,
    apply_projection,
)

BASE_MODEL_ID = "gemma-test-object"
RUNTIME_PROFILE_ID = "mlx-test-profile"
RULE_PROFILE_ID = "tensor-logic-page-selection-v1"
WEIGHT = ((0.0, 0.0), (0.0, 0.0))
INPUT = (1.0, 0.0)


def _page(domain: str, direction: float, version: int = 1) -> ParameterPage:
    return ParameterPage(
        base_model_id=BASE_MODEL_ID,
        runtime_profile_id=RUNTIME_PROFILE_ID,
        training_recipe_id="frozen-recipe",
        evidence_closure_id=f"{domain}-evidence-{version}",
        privacy_domain_id=domain,
        page_version=version,
        sites=(
            SitePage(
                site_name="classifier",
                input_width=2,
                output_width=2,
                rank=1,
                scale=1.0,
                matrix_a=((1.0, 0.0),),
                matrix_b=((direction,), (-direction,)),
            ),
        ),
    )


def _request(principal: str, domain: str, epoch: int = 1) -> ActivationRequest:
    return ActivationRequest(
        invocation_id=f"invocation-{principal}-{epoch}",
        principal_id=principal,
        privacy_domain_id=domain,
        concept="kelthara",
        epoch=epoch,
        base_model_id=BASE_MODEL_ID,
        runtime_profile_id=RUNTIME_PROFILE_ID,
        site_budgets=(("classifier", 2.0),),
    )


def _selector(
    pages: tuple[ParameterPage, ...],
    memberships: tuple[DomainMembership, ...],
    grants: tuple[PageGrant, ...],
    services: tuple[PageService, ...],
) -> TensorLogicPageSelector:
    return TensorLogicPageSelector(
        pages,
        memberships,
        grants,
        services,
        rule_profile_id=RULE_PROFILE_ID,
    )


def _bits(values: tuple[float, ...]) -> bytes:
    return b"".join(struct.pack(">d", value) for value in values)


def test_zero_page_path_is_bit_identical_to_base_projection() -> None:
    selector = _selector((), (), (), ())
    proof = selector.select(_request("alice", "alpha"))

    actual = apply_projection(
        WEIGHT, INPUT, site_name="classifier", proof=proof, pages=()
    )
    expected = apply_projection(
        WEIGHT,
        INPUT,
        site_name="classifier",
        proof=selector.select(_request("nobody", "none")),
        pages=(),
    )

    assert _bits(actual) == _bits(expected)


def test_tensor_logic_selection_isolates_contradictory_domains() -> None:
    alice_page = _page("alpha", 1.0)
    bob_page = _page("beta", -1.0)
    pages = (alice_page, bob_page)
    selector = _selector(
        pages,
        (
            DomainMembership("alice", "alpha"),
            DomainMembership("bob", "beta"),
        ),
        (
            PageGrant("alice", alice_page.page_id),
            PageGrant("bob", bob_page.page_id),
        ),
        (
            PageService(alice_page.page_id, "kelthara", 0, 10),
            PageService(bob_page.page_id, "kelthara", 0, 10),
        ),
    )

    alice_proof = selector.select(_request("alice", "alpha"))
    bob_proof = selector.select(_request("bob", "beta"))
    alice_output = apply_projection(
        WEIGHT,
        INPUT,
        site_name="classifier",
        proof=alice_proof,
        pages=pages,
    )
    bob_output = apply_projection(
        WEIGHT, INPUT, site_name="classifier", proof=bob_proof, pages=pages
    )

    assert alice_proof.selected_page_ids == (alice_page.page_id,)
    assert bob_proof.selected_page_ids == (bob_page.page_id,)
    assert alice_output == (1.0, -1.0)
    assert bob_output == (-1.0, 1.0)


def test_epoch_selects_immutable_replacement_and_restart_is_stable() -> None:
    old_page = _page("alpha", 1.0, version=1)
    new_page = _page("alpha", -1.0, version=2)
    pages = (old_page, new_page)
    memberships = (DomainMembership("alice", "alpha"),)
    grants = (
        PageGrant("alice", old_page.page_id),
        PageGrant("alice", new_page.page_id),
    )
    services = (
        PageService(old_page.page_id, "kelthara", 0, 1),
        PageService(new_page.page_id, "kelthara", 2, 10),
    )

    first_selector = _selector(pages, memberships, grants, services)
    restarted_selector = _selector(pages, memberships, grants, services)
    old_proof = first_selector.select(_request("alice", "alpha", epoch=1))
    new_proof = first_selector.select(_request("alice", "alpha", epoch=2))
    restarted_proof = restarted_selector.select(_request("alice", "alpha", epoch=2))

    assert old_proof.selected_page_ids == (old_page.page_id,)
    assert new_proof.selected_page_ids == (new_page.page_id,)
    assert new_proof.proof_id == restarted_proof.proof_id


def test_revocation_restores_the_exact_base_path() -> None:
    page = _page("alpha", 1.0)
    membership = (DomainMembership("alice", "alpha"),)
    service = (PageService(page.page_id, "kelthara", 0, 10),)
    mounted = _selector(
        (page,), membership, (PageGrant("alice", page.page_id),), service
    ).select(_request("alice", "alpha"))
    revoked = _selector((page,), membership, (), service).select(
        _request("alice", "alpha")
    )

    mounted_output = apply_projection(
        WEIGHT, INPUT, site_name="classifier", proof=mounted, pages=(page,)
    )
    revoked_output = apply_projection(
        WEIGHT, INPUT, site_name="classifier", proof=revoked, pages=(page,)
    )
    base_output = (0.0, 0.0)

    assert mounted_output != base_output
    assert revoked.selected_page_ids == ()
    assert _bits(revoked_output) == _bits(base_output)


def test_conflicting_pages_fail_closed() -> None:
    first = _page("alpha", 1.0, version=1)
    second = _page("alpha", -1.0, version=2)
    selector = _selector(
        (first, second),
        (DomainMembership("alice", "alpha"),),
        (PageGrant("alice", first.page_id), PageGrant("alice", second.page_id)),
        (
            PageService(first.page_id, "kelthara", 0, 10),
            PageService(second.page_id, "kelthara", 0, 10),
        ),
    )

    with pytest.raises(AmbiguousActivationError):
        selector.select(_request("alice", "alpha"))


def test_perturbation_budget_fails_closed() -> None:
    page = _page("alpha", 3.0)
    selector = _selector(
        (page,),
        (DomainMembership("alice", "alpha"),),
        (PageGrant("alice", page.page_id),),
        (PageService(page.page_id, "kelthara", 0, 10),),
    )
    request = ActivationRequest(
        invocation_id="budget-test",
        principal_id="alice",
        privacy_domain_id="alpha",
        concept="kelthara",
        epoch=1,
        base_model_id=BASE_MODEL_ID,
        runtime_profile_id=RUNTIME_PROFILE_ID,
        site_budgets=(("classifier", 1.0),),
    )

    with pytest.raises(PerturbationBudgetExceededError):
        selector.select(request)
