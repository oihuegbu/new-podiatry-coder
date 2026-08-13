"""Shared, implementation-neutral contracts.

Nothing in this package may import a pipeline implementation (`app.pipeline`,
`claude_coder.*`). These modules are the vocabulary those implementations and
their downstream consumers (claims registry, readiness verification, 837P
construction) agree on; if a contract had to import one producer to be
understood, it would belong to that producer and the next producer would drift
away from it again — which is exactly the defect this package exists to close
(issue #6, finding F6-R4-A1).
"""
