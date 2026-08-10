"""Who someone is, and which organizations they belong to.

Kept apart from `MemoryService` because it answers a different question. That
service answers "what does this organization remember"; this one answers "who
is asking, and are they allowed to ask on that organization's behalf".

The distinction that matters, and the reason this exists at all:

    an API KEY authorises a request against an organization
    an EMAIL identifies a person

They are not interchangeable. Two people on one team legitimately share one
key, so a key can never tell you who acted; and revoking a person's access
should not revoke a running service's. Every audit trail that conflates them
answers "which credential did this" when the question was "who did this".
"""

from __future__ import annotations

from dataclasses import dataclass

from .core.errors import ConflictError, ForbiddenError, NotFoundError
from .domain.models import MemberRole, Membership, Organization, User, utcnow
from .store.base import MemoryStore

#: Organizations one person may belong to. A product limit, not a technical
#: one -- the uniqueness constraint in the schema holds whatever this is.
#: Deliberately small: this is a personal memory product, and an account that
#: can fan out indefinitely is an account worth farming.
MAX_ORGS_PER_USER = 3


@dataclass(frozen=True, slots=True)
class OrgSummary:
    """An organization as it appears in someone's own list."""

    organization: Organization
    role: MemberRole
    spaces: int


class IdentityService:
    def __init__(self, store: MemoryStore) -> None:
        self.store = store

    async def sign_in_with_google(self, claims: dict[str, object]) -> User:
        """Create or refresh the user behind a verified Google token.

        Matches on `sub`, never on email. Google's subject id is stable for
        the life of the account; an email can be reassigned to someone else
        inside a workspace, and matching on it would hand the new holder the
        previous person's memories.
        """
        sub = str(claims.get("sub") or "").strip()
        email = str(claims.get("email") or "").strip()
        if not sub or not email:
            raise ForbiddenError("Google did not return a subject and email")
        if claims.get("email_verified") is False:
            # An unverified address is a claim about an inbox nobody checked.
            raise ForbiddenError("this Google account has an unverified email")

        return await self.store.upsert_user(
            User(
                email=email,
                google_sub=sub,
                name=str(claims.get("name") or ""),
                picture=str(claims.get("picture") or ""),
                last_seen_at=utcnow(),
            )
        )

    async def create_organization(self, user: User, name: str) -> Organization:
        """Create an organization and make the creator its owner.

        The cap is checked here rather than in the store because it is a
        product decision; the store's job is that a person cannot join the
        same organization twice, which is true under any cap.
        """
        existing = await self.store.list_memberships(user.id)
        if len(existing) >= MAX_ORGS_PER_USER:
            raise ConflictError(
                f"an account may belong to at most {MAX_ORGS_PER_USER} organizations; "
                f"leave one before creating another"
            )
        org = await self.store.create_organization(Organization(name=name.strip()))
        await self.store.create_membership(
            Membership(user_id=user.id, org_id=org.id, role=MemberRole.OWNER)
        )
        return org

    async def list_organizations(self, user: User) -> list[OrgSummary]:
        """Every organization this person belongs to, oldest membership first."""
        out: list[OrgSummary] = []
        for membership in await self.store.list_memberships(user.id):
            org = await self.store.get_organization(membership.org_id)
            if org is None:
                # A membership outliving its organization is not something to
                # surface to a person; it is a cleanup problem.
                continue
            spaces = await self.store.list_spaces(org.id)
            out.append(OrgSummary(organization=org, role=membership.role, spaces=len(spaces)))
        return out

    async def require_member(self, user: User, org_id: str) -> Membership:
        """Assert this person belongs to that organization.

        Raises NotFound rather than Forbidden for a non-member, matching the
        rule the rest of the API follows: a 403 confirms the organization
        exists, which leaks it to anyone who can guess an id.
        """
        membership = await self.store.get_membership(user.id, org_id)
        if membership is None:
            raise NotFoundError(f"organization {org_id} not found", field="org_id")
        return membership


__all__ = ["MAX_ORGS_PER_USER", "IdentityService", "OrgSummary"]
