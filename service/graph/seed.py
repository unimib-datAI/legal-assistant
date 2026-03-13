"""
SEEDS_V3 — Revised ASKE seed for Legal KG v1.1
================================================
Designed to address cross-act topic pollution observed in SEEDS_AI_DATA_FOCUSED_v2.

Design principles:
  P1 — Act-anchored phrasing (no bare terms shared across acts)
  P2 — Principle-complete for GDPR Art. 5
  P3 — Minimum 3-word seeds
  P4 — No duplicate semantic space
  P5 — No bare domain labels
  P6 — Act-structured layers
  P7 — Cross-cutting themes are last-resort

Total seeds: 97
"""

SEEDS = [
    # ═══════════════════════════════════════════════════════════════
    # LAYER 1 — GDPR CORE PRINCIPLES (Art. 5)
    # These are the foundation. Each must survive as a distinct topic
    # because RAG queries targeting GDPR principles MUST retrieve
    # GDPR paragraphs, not semantically similar text from other acts.
    # ═══════════════════════════════════════════════════════════════
    "lawfulness fairness and transparency of processing",
    "purpose limitation principle",
    "data minimisation principle",
    "data accuracy obligation",
    "storage limitation principle",
    "integrity and confidentiality of processing",
    "accountability principle",

    # ═══════════════════════════════════════════════════════════════
    # LAYER 2 — GDPR LEGAL BASES & CONSENT (Art. 6-10)
    # ═══════════════════════════════════════════════════════════════
    "lawfulness of processing",
    "consent as legal basis",
    "legitimate interest assessment",
    "processing of special category data",
    "processing of criminal offence data",

    # ═══════════════════════════════════════════════════════════════
    # LAYER 3 — GDPR DATA SUBJECT RIGHTS (Art. 12-23)
    # ═══════════════════════════════════════════════════════════════
    "right of access by the data subject",
    "right to rectification",
    "right to erasure",
    "right to restriction of processing",
    "right to data portability",
    "right to object to processing",
    "automated individual decision-making and profiling",
    "transparent information and communication",

    # ═══════════════════════════════════════════════════════════════
    # LAYER 4 — GDPR CONTROLLER & PROCESSOR (Art. 24-31)
    # ═══════════════════════════════════════════════════════════════
    "controller responsibility and accountability",
    "data protection by design and by default",
    "joint controller arrangement",
    "processor obligation and contract",
    "records of processing activities",

    # ═══════════════════════════════════════════════════════════════
    # LAYER 5 — GDPR SECURITY & BREACH (Art. 32-34)
    # ═══════════════════════════════════════════════════════════════
    "security of processing",
    "personal data breach notification",
    "communication of breach to data subject",

    # ═══════════════════════════════════════════════════════════════
    # LAYER 6 — GDPR IMPACT ASSESSMENT & DPO (Art. 35-39)
    # ═══════════════════════════════════════════════════════════════
    "data protection impact assessment",
    "prior consultation with supervisory authority",
    "data protection officer designation",
    "data protection officer tasks and independence",

    # ═══════════════════════════════════════════════════════════════
    # LAYER 7 — GDPR INTERNATIONAL TRANSFERS (Art. 44-49)
    # ═══════════════════════════════════════════════════════════════
    "international data transfer safeguard",
    "adequacy decision for data transfer",
    "standard contractual clauses for transfer",
    "binding corporate rules",
    "transfer impact assessment",

    # ═══════════════════════════════════════════════════════════════
    # LAYER 8 — GDPR SUPERVISION & ENFORCEMENT (Art. 51-84)
    # ═══════════════════════════════════════════════════════════════
    "supervisory authority powers and tasks",
    "lead supervisory authority and one-stop-shop",
    "right to judicial remedy",
    "right to compensation for data protection violation",
    "administrative fine and penalty",
    "cross-border processing cooperation",

    # ═══════════════════════════════════════════════════════════════
    # LAYER 9 — AI ACT RISK FRAMEWORK (Reg. 2024/1689)
    # ═══════════════════════════════════════════════════════════════
    "prohibited artificial intelligence practice",
    "high-risk AI system classification",
    "AI risk management system",
    "high-risk AI data governance requirement",
    "high-risk AI technical documentation",
    "high-risk AI transparency requirement",
    "high-risk AI human oversight",
    "high-risk AI accuracy robustness and cybersecurity",
    "AI conformity assessment procedure",
    "AI post-market monitoring obligation",

    # ═══════════════════════════════════════════════════════════════
    # LAYER 10 — AI ACT ACTORS & GOVERNANCE
    # ═══════════════════════════════════════════════════════════════
    "AI provider obligation",
    "AI deployer obligation",
    "AI importer and distributor obligation",
    "general-purpose AI model obligation",
    "general-purpose AI systemic risk",
    "AI regulatory sandbox",
    "real-world AI testing condition",
    "AI literacy obligation",
    "CE marking for AI system",
    "notified body for AI conformity",

    # ═══════════════════════════════════════════════════════════════
    # LAYER 11 — AI ACT SPECIFIC APPLICATIONS
    # ═══════════════════════════════════════════════════════════════
    "biometric identification and categorisation",
    "emotion recognition system",
    "AI in critical infrastructure",
    "AI in law enforcement context",
    "AI in migration and border control",
    "AI in administration of justice",
    "AI in employment and worker management",
    "AI in education and vocational training",

    # ═══════════════════════════════════════════════════════════════
    # LAYER 12 — DATA ACT (Reg. 2023/2854)
    # ═══════════════════════════════════════════════════════════════
    "data access right for connected product user",
    "data holder obligation under Data Act",
    "data recipient obligation and use limitation",
    "data sharing for connected product and service",
    "unfair contractual term in data sharing",
    "trade secret protection in data access",
    "IoT data and connected device",
    "cloud switching and interoperability",
    "data made available to public sector body",
    "smart contract for data sharing",

    # ═══════════════════════════════════════════════════════════════
    # LAYER 13 — DATA GOVERNANCE ACT (Reg. 2022/868)
    # ═══════════════════════════════════════════════════════════════
    "re-use of protected public sector data",
    "data intermediation service provision",
    "data altruism organisation and registration",
    "data cooperative governance",
    "European data innovation board",
    "consent and permission management for data sharing",
    "single information point for data re-use",
    "competent body for data re-use assistance",

    # ═══════════════════════════════════════════════════════════════
    # LAYER 14 — CROSS-CUTTING THEMES
    # These exist across multiple acts but must remain distinct from
    # act-specific seeds. Phrased to be discriminative.
    # ═══════════════════════════════════════════════════════════════
    "fundamental rights impact assessment",
    "market surveillance for regulated product",
    "harmonised standard and standardisation",
    "incident reporting obligation",
    "redress and complaint mechanism",
    "auditability and traceability requirement",
    "certification and code of conduct",
]