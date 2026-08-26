import { collections } from "./db.js";

// The invited-participant roster. The study is distributed to a hand-picked set
// of Prolific IDs (see scripts/seed_roster.py), each of whom told us their job
// area and job title in the screener app. Prolific's own participant allowlist
// is the primary gate; this is the second one, so a shared link cannot pull in
// somebody we have no screener answers for, and it is also the lookup table
// that lets the landing page SHOW a participant their area and title rather
// than ask for them again.
//
// It lives in MongoDB, not in a file in the repo, for two reasons. Prolific IDs
// are personal data and the repo has a GitHub remote, so a committed table
// would publish them. And the roster changes during recruitment: swapping a
// participant out, or topping an under-filled job area up from the reserve
// lists, is a re-run of the seed script rather than a redeploy.
//
// Prolific IDs are case-stable in practice, but participants who type theirs by
// hand get the case wrong often enough to be worth absorbing here. The
// collection is keyed lower-case, so we normalise on the way in.
export async function lookupRoster(prolificId) {
  const pid = String(prolificId ?? "").trim().toLowerCase();
  if (!pid) return null;
  return collections.roster().findOne(
    { prolificId: pid },
    { projection: { _id: 0, cluster: 1, jobTitle: 1 } }
  );
}

// The role the job-specific study assigned this participant (its
// `recipientRole`, e.g. "Executive Assistant"). They keep it here: judging the
// generic emails as the same role holds the recipient's standing constant
// across the two sets, so "could a sender at that level send this to someone at
// my level" means the same thing in both, and the ratings stay comparable.
//
// Read from that study's participants collection, which is the authoritative
// record of what this person was actually shown. It is one role per job area,
// so the client can fall back to the roles map in content.json (keyed by
// cluster) if this returns nothing, which is what happens for anyone invited
// here who never finished the job-specific study.
export async function lookupAssignedRole(prolificId) {
  const pid = String(prolificId ?? "").trim().toLowerCase();
  if (!pid) return null;
  const row = await collections.jobSpecificParticipants().findOne(
    { prolificId: pid },
    { projection: { _id: 0, recipientRole: 1 } }
  );
  return (row && row.recipientRole) || null;
}
