from app.models.schemas import ClinicalEntity
from app.rag.vector_store import MedicalCodeVectorStore
from app.core.config import RAG_TOP_K
from app.core.logger import get_logger

logger = get_logger(__name__)

CATEGORY_TO_CODE_SYSTEMS = {
    "diagnosis": ["icd10"],
    "finding": ["icd10"],
    "procedure": ["cpt"],
    # Medications retrieve HCPCS candidates: drugs ADMINISTERED in-office
    # (injectables) are separately billable J-codes, and the J-code family
    # lives in the HCPCS collection. Previously mapped to [] — no candidates
    # were ever retrieved for any drug, so J-code selection depended entirely
    # on the prompt's memorized examples; any administered drug outside that
    # list was silently missed (unbilled) or guessed and then removed by the
    # hard DB gate. Oral/topical prescriptions retrieving irrelevant J-code
    # candidates is harmless — the prompts already restrict J-codes to drugs
    # administered today, and candidates are advisory, not assignments.
    "medication": ["hcpcs"],
    "supply": ["hcpcs"],
    "body_structure": [],
    # "allergy" is intentionally absent: it falls through to the
    # ["icd10", "cpt"] default below, which retrieves the Z88.x drug-allergy
    # status codes from the ICD-10 collection.
}


class CandidateRetriever:
    """Retrieves candidate medical codes for clinical entities via Qdrant hybrid search."""

    def __init__(self, vector_store: MedicalCodeVectorStore):
        self.store = vector_store

    def retrieve_for_entity(
        self,
        entity: ClinicalEntity,
        top_k: int | None = None,
    ) -> dict[str, list[dict]]:
        top_k = top_k or RAG_TOP_K

        code_systems = CATEGORY_TO_CODE_SYSTEMS.get(entity.category, ["icd10", "cpt"])
        if not code_systems:
            return {}

        results = {}

        for cs in code_systems:
            per_query = [self.store.search(query, cs, top_k=top_k)
                         for query in self._build_queries(entity)]
            # Each query form gets equal rank opportunity.  A mistaken model
            # normalization can no longer crowd out the verbatim phrase or a
            # governed deterministic expansion before the coder sees it.
            candidates = self._round_robin(per_query, top_k)
            if candidates:
                results[cs] = candidates

        return results

    def retrieve_for_entities(
        self,
        entities: list[ClinicalEntity],
        top_k: int | None = None,
    ) -> dict[str, dict[str, list[dict]]]:
        all_results = {}
        for index, entity in enumerate(entities):
            # Include ordinal + verbatim span so repeated entities with the
            # same normalized term cannot overwrite one another.
            key = f"{index}:{entity.category}:{entity.clinical_term}:{entity.text}"
            candidates = self.retrieve_for_entity(entity, top_k)
            if candidates:
                all_results[key] = {
                    "entity": entity.model_dump(),
                    "candidates": candidates,
                }
        return all_results

    def retrieve_for_full_note(
        self,
        note_sections: dict,
        top_k: int | None = None,
    ) -> dict[str, list[dict]]:
        top_k = top_k or RAG_TOP_K

        assessment = note_sections.get("assessment_diagnoses", "")
        plan = note_sections.get("plan", "")
        pmh = note_sections.get("pmh_medications_allergies", "")
        imaging = note_sections.get("imaging_diagnostics", "")

        icd_queries = [
            f"Podiatry diagnosis: {assessment}",
            f"Chronic medical conditions: {pmh}",
        ]

        cpt_queries = [
            f"Podiatry surgical procedure: {plan}",
        ]
        if imaging and imaging.strip().lower() not in ("none indicated.", "n/a", "none", ""):
            cpt_queries.append(f"Radiology imaging X-ray foot ankle: {imaging}")

        # Two distinct HCPCS families live in the same collection and don't
        # co-retrieve well from one query: supplies/DME (A/L/E-codes) and
        # injectable drugs (J-codes). A supply-worded query alone never
        # surfaced J-code candidates, so in-office injections (e.g.
        # corticosteroid for plantar fasciitis) had no drug candidates at all.
        hcpcs_queries = [
            f"Medical supply DME boot shoe orthotic dressing: {plan}",
            f"Injectable drug injection medication administered: {plan}",
        ]

        icd_candidates = self._round_robin(
            [self.store.search(q, "icd10", top_k=top_k) for q in icd_queries],
            top_k)
        cpt_candidates = self._round_robin(
            [self.store.search(q, "cpt", top_k=top_k) for q in cpt_queries],
            top_k)
        hcpcs_candidates = self._round_robin(
            [self.store.search(q, "hcpcs", top_k=top_k) for q in hcpcs_queries],
            top_k)

        return {
            "icd10": icd_candidates,
            "cpt": cpt_candidates,
            "hcpcs": hcpcs_candidates,
        }

    def _build_queries(self, entity: ClinicalEntity) -> list[str]:
        base_terms = entity.retrieval_terms or [entity.clinical_term, entity.text]
        queries = []
        seen = set()
        for term in base_terms:
            parts = [str(term or "").strip()]
            if entity.laterality:
                parts.append(entity.laterality.lower())
            if entity.specificity:
                parts.append(entity.specificity)
            query = " ".join(part for part in parts if part).strip()
            key = query.casefold()
            if query and key not in seen:
                seen.add(key)
                queries.append(query)
        return queries

    @staticmethod
    def _round_robin(per_query: list[list[dict]], limit: int) -> list[dict]:
        """Fuse separately ranked queries without comparing their scores."""
        candidates = []
        seen: set[str] = set()
        depth = 0
        while len(candidates) < limit and any(
                depth < len(rows) for rows in per_query):
            for rows in per_query:
                if depth >= len(rows):
                    continue
                candidate = dict(rows[depth])
                code = str(candidate.get("code") or "").strip()
                if code and code not in seen:
                    seen.add(code)
                    candidates.append(candidate)
                    if len(candidates) >= limit:
                        break
            depth += 1
        return candidates

    def _build_query(self, entity: ClinicalEntity) -> str:
        """Backward-compatible primary query; multi-form retrieval uses all."""
        queries = self._build_queries(entity)
        return queries[0] if queries else ""
