"""Agent registry. `build_default_agents` returns the active set of compliance
agents in filter order. Agents are added here as each phase lands."""
from __future__ import annotations

from app.compliance.datastore.store import ComplianceDataStore
from app.compliance.agents.base import ComplianceAgent


def build_default_agents(store: ComplianceDataStore) -> list[ComplianceAgent]:
    from app.compliance.agents.specificity import SpecificityAgent       # #1
    from app.compliance.agents.ncci_ptp import NCCIPTPAgent               # #2
    from app.compliance.agents.mue_mai import MUEAgent                    # #3
    from app.compliance.agents.modifiers import ModifierAgent             # #4
    from app.compliance.agents.medical_necessity import MedicalNecessityAgent  # #5
    from app.compliance.agents.global_period import GlobalPeriodAgent     # #6
    from app.compliance.agents.frequency import FrequencyAgent            # #7
    from app.compliance.agents.addon import AddOnAgent                    # #8
    from app.compliance.agents.pos_eligibility import POSEligibilityAgent # #9
    from app.compliance.agents.prior_auth import PriorAuthAgent           # #10
    from app.compliance.agents.benefits import BenefitsAgent              # #11
    from app.compliance.agents.documentation import DocumentationAgent    # #12
    from app.compliance.agents.billability import BillabilityAgent        # #13
    from app.compliance.agents.mce import MCEAgent                        # #14
    from app.compliance.agents.surgical_package import SurgicalPackageAgent  # #15

    agents: list[ComplianceAgent] = [
        SpecificityAgent(store),       # 1
        NCCIPTPAgent(store),           # 2
        MUEAgent(store),               # 3
        ModifierAgent(store),          # 4
        MedicalNecessityAgent(store),  # 5
        GlobalPeriodAgent(store),      # 6
        FrequencyAgent(store),         # 7
        AddOnAgent(store),             # 8
        POSEligibilityAgent(store),    # 9
        PriorAuthAgent(store),         # 10
        BenefitsAgent(store),          # 11
        DocumentationAgent(store),     # 12
        BillabilityAgent(store),       # 13
        MCEAgent(store),               # 14
        SurgicalPackageAgent(store),   # 15
    ]
    return agents
