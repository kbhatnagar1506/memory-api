"""Identity: who is asking, and may they ask for that organization.

The rule these tests encode is that an API key AUTHORISES and an email
IDENTIFIES, and the two are not interchangeable.
"""

from __future__ import annotations

import pytest

from mapi.core.errors import ConflictError, ForbiddenError, NotFoundError
from mapi.domain.models import MemberRole
from mapi.identity import MAX_ORGS_PER_USER, IdentityService
from mapi.store.memory import InMemoryStore

CLAIMS = {
    "sub": "108813",
    "email": "Krishna@Profitwise.App",
    "email_verified": True,
    "name": "Krishna",
    "picture": "https://example.test/a.png",
}


@pytest.fixture
def identity() -> IdentityService:
    return IdentityService(InMemoryStore())


async def test_sign_in_creates_a_user_with_a_normalized_email(identity) -> None:
    user = await identity.sign_in_with_google(CLAIMS)
    assert user.email == "krishna@profitwise.app"
    assert user.google_sub == "108813"


async def test_signing_in_again_keeps_the_same_user_id(identity) -> None:
    """Memberships point at the id; a new one per login would orphan them."""
    first = await identity.sign_in_with_google(CLAIMS)
    again = await identity.sign_in_with_google({**CLAIMS, "name": "Krishna B"})
    assert again.id == first.id
    assert again.name == "Krishna B"


async def test_identity_follows_the_subject_not_the_email(identity) -> None:
    """An email can be reassigned inside a workspace; the subject cannot.
    Matching on email would hand the new holder the old person's memories."""
    original = await identity.sign_in_with_google(CLAIMS)
    reassigned = await identity.sign_in_with_google(
        {**CLAIMS, "sub": "999999", "email": CLAIMS["email"]}
    )
    assert reassigned.id != original.id


async def test_an_unverified_email_cannot_sign_in(identity) -> None:
    with pytest.raises(ForbiddenError):
        await identity.sign_in_with_google({**CLAIMS, "email_verified": False})


async def test_missing_claims_are_refused(identity) -> None:
    for bad in ({"email": "a@b.c"}, {"sub": "1"}, {}):
        with pytest.raises(ForbiddenError):
            await identity.sign_in_with_google(bad)


async def test_the_creator_becomes_the_owner(identity) -> None:
    user = await identity.sign_in_with_google(CLAIMS)
    org = await identity.create_organization(user, "Acme")
    summaries = await identity.list_organizations(user)
    assert [s.organization.id for s in summaries] == [org.id]
    assert summaries[0].role is MemberRole.OWNER


async def test_the_org_cap_is_enforced(identity) -> None:
    user = await identity.sign_in_with_google(CLAIMS)
    for i in range(MAX_ORGS_PER_USER):
        await identity.create_organization(user, f"Org {i}")
    with pytest.raises(ConflictError, match="at most"):
        await identity.create_organization(user, "One too many")


async def test_the_cap_is_per_user_not_global(identity) -> None:
    a = await identity.sign_in_with_google(CLAIMS)
    b = await identity.sign_in_with_google({**CLAIMS, "sub": "2", "email": "b@x.test"})
    for i in range(MAX_ORGS_PER_USER):
        await identity.create_organization(a, f"A{i}")
    org = await identity.create_organization(b, "B0")
    assert org.id


async def test_a_non_member_is_told_the_org_does_not_exist(identity) -> None:
    """404 not 403: a 403 confirms the organization exists, which leaks it to
    anyone who can guess an id."""
    owner = await identity.sign_in_with_google(CLAIMS)
    org = await identity.create_organization(owner, "Private")
    outsider = await identity.sign_in_with_google(
        {**CLAIMS, "sub": "outsider", "email": "o@x.test"}
    )
    with pytest.raises(NotFoundError):
        await identity.require_member(outsider, org.id)
    assert await identity.require_member(owner, org.id)


async def test_one_person_cannot_join_the_same_org_twice(identity) -> None:
    from mapi.domain.models import Membership

    user = await identity.sign_in_with_google(CLAIMS)
    org = await identity.create_organization(user, "Acme")
    with pytest.raises(ConflictError):
        await identity.store.create_membership(Membership(user_id=user.id, org_id=org.id))
