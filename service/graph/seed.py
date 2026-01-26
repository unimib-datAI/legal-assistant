SEEDS = [
    "political framework",
    "political party",
    "electoral procedure and voting",
    "parliament",
    "parliamentary proceedings",
    "politics and public safety",
    "international affairs",
    "cooperation policy",
    "international balance",
    "defence",
    "European construction",
    "European Union law",
    "EU institutions and European civil service",
    "EU finance",
    "EU policies",
    "sources and branches of law",
    "civil law",
    "criminal law",
    "justice",
    "organisation of the legal system",
    "international law",
    "rights and freedoms",
    "economic policy",
    "economic structure",
    "regions and regional policy",
    "economic geography",
    "economic analysis",
    "economic statistics",
    "accounting",
    "trade policy",
    "tariff policy",
    "trade",
    "international trade",
    "consumption",
    "marketing",
    "distributive trades",
    "monetary economics",
    "financial institutions and credit",
    "free movement of capital",
    "financing and investment",
    "public finance and budget policy",
    "budget",
    "taxation",
    "prices",
    "insurance",
    "family",
    "migration",
    "demography and population",
    "social framework",
    "social affairs",
    "culture and religion",
    "social protection",
    "health",
    "urban development and urban policy",
    "education",
    "teaching",
    "organisation of teaching",
    "documentation",
    "communications",
    "information and information processing",
    "information technology and data processing",
    "natural and applied sciences",
    "humanities",
    "business organisation",
    "business classification",
    "legal form of organisations",
    "management",
    "accounting, commerce and company law",
    "competition",
    "employment",
    "labour market",
    "organisation of work and working conditions",
    "personnel management and staff remuneration",
    "labour law and labour relations",
    "transport policy",
    "organisation of transport",
    "land transport",
    "maritime and inland waterway transport",
    "air and space transport",
    "environmental policy",
    "natural environment",
    "deterioration of the environment",
    "agricultural policy",
    "agricultural structures and production",
    "farming systems",
    "cultivation of agricultural land",
    "means of agricultural production",
    "agricultural activity",
    "forestry",
    "fisheries",
    "plant product",
    "animal product",
    "processed agricultural produce",
    "beverages and sugar",
    "foodstuff",
    "soft drink",
    "production",
    "technology and technical regulations",
    "research and intellectual property",
    "energy policy",
    "coal and mining industries",
    "oil industry",
    "electrical and nuclear industries",
    "soft energy",
    "industrial structures and policy",
    "chemistry",
    "iron, steel and other metal industries",
    "mechanical engineering",
    "electronics and electrical engineering",
    "building and public works",
    "wood industry",
    "leather and textile industries",
    "miscellaneous industries",
    "Europe",
    "regions of EU Member States",
    "America",
    "Africa",
    "Asia and Oceania",
    "political geography",
    "overseas countries and territories",
    "United Nations",
    "European organisations",
    "extra-European organisations",
    "world organisations"
]

SEEDS_AI_DATA_FOCUSED = [
    # === CORE AI/DATA CONCEPTS ===
    "artificial intelligence",
    "machine learning",
    "algorithmic decision-making",
    "automated decision-making",
    "AI system",
    "high-risk AI system",
    
    # === DATA GOVERNANCE ===
    "data protection",
    "personal data",
    "data processing",
    "data subject",
    "data controller",
    "data processor",
    "data governance",
    "data sharing",
    "data intermediary",
    "data altruism",
    
    # === RIGHTS & OBLIGATIONS ===
    "rights and freedoms",
    "fundamental rights",
    "privacy rights",
    "right to explanation",
    "transparency obligation",
    "accountability",
    
    # === TECHNICAL/OPERATIONAL ===
    "technical documentation",
    "conformity assessment",
    "risk management",
    "human oversight",
    "accuracy requirement",
    "cybersecurity",
    "data quality",
    
    # === REGULATORY ===
    "supervisory authority",
    "enforcement",
    "compliance",
    "sanctions",
    "certification",
    
    # === SPECIFIC CONTEXTS ===
    "biometric identification",
    "emotion recognition",
    "critical infrastructure"
]

ENHANCED_SEEDS = [
    # === Keep all current specific seeds (35) ===
    "artificial intelligence",
    "machine learning",
    "data protection",
    # ... all current seeds

    # === Add 8-10 TARGETED generic seeds ===
    # To address underdeveloped concepts

    # For "data processing" (1 term)
    "processing",  # Generic but relevant

    # For "AI system" (1 term)
    "system",  # Will capture AI + management + technical systems

    # For "automated decision-making" (1 term)
    "decision",  # Will capture decision types

    # For "data controller" (1 term)
    "controller",  # Will capture controller types

    # For missing concepts
    "consent",  # Critical GDPR concept
    "profiling",  # Already in [objectivity] but deserves own seed

    # To enrich existing
    "algorithm",  # Complement to "algorithmic decision-making"
    "assessment",  # Impact assessment, conformity assessment

    # Legal process terms
    "obligation",  # Legal obligations
    "right",  # Data subject rights (already derived, reinforce)
]