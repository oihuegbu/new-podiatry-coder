# Medium-sized private surgical-practice scope

The initial autonomy lane targets professional surgical claims produced by a
declared medium-sized private practice. Scope is evaluated from versioned
configuration, not Python conditionals.

An encounter is eligible only when the organizational and surgical profiles,
professional claim type, supported facility context, payer identity, contract
profile, billing entity, performing entity, jurisdiction, place of service,
and authorization status are present.

Institutional claims and unknown organizational, payer, or facility contexts
are outside this lane. They receive an explicit non-release disposition rather
than borrowing professional-claim policy.

This scope does not assert that every surgical encounter is automatically
releasable. Every in-scope encounter must still pass source identity, date
validity, descriptor entailment, specificity, PTP, add-on, units, claim
context, coverage, provenance, and other configured gates.

