"""Prompts for the offline article and chapter summarisation pipelines."""
from datetime import date

from legal_assistant.rag.prompts.registry import PromptVersion, registry


ARTICLE_SUMMARY_SYSTEM_V1 = """You are an expert in EU digital legislation (GDPR, AI Act, Data Act, Data Governance Act).
Produce concise retrieval-optimised summaries of legal articles so that semantic similarity search
can correctly match user questions to the right article. Use plain, query-friendly language.
Follow the output format exactly — do not add extra sections or omit any field."""

ARTICLE_SUMMARY_USER_V1 = """Summarise the following legal article using the template below.

=== OUTPUT TEMPLATE ===
Act: {{regulation_name}} ({{celex}})
Article — {{title}}
Applies to: {{entities subject to this article}}
Core rule: {{main obligation / right / prohibition in plain language}}
Key conditions or exceptions: {{when it applies / when it does not}}
Procedure: {{numbered steps if the article is procedural, otherwise "N/A"}}
Key terms: {{defined or heavily-used legal terms, comma-separated}}
Cross-references: {{articles explicitly mentioned in the text, or "none"}}

=== FEW-SHOT EXAMPLES ===

--- Example 1 ---
INPUT
Act: General Data Protection Regulation (32016R0679) | Article — Right to erasure ('right to be forgotten')
Body:
1. The data subject shall have the right to obtain from the controller the erasure of personal data
concerning him or her without undue delay [...] where one of the following grounds applies:
(a) the personal data are no longer necessary in relation to the purposes for which they were collected;
(b) the data subject withdraws consent [...]; (c) the data subject objects [...]; (d) the personal data
have been unlawfully processed; (e) erasure required by Union or Member State law; (f) data collected
in relation to the offer of information society services referred to in Article 8(1).
2. Where the controller has made the personal data public [...] shall take reasonable steps to inform
controllers which are processing the personal data that the data subject has requested the erasure.
3. Paragraphs 1 and 2 shall not apply to the extent that processing is necessary: (a) for exercising
the right of freedom of expression and information; (b) for compliance with a legal obligation [...]

OUTPUT
Act: General Data Protection Regulation (32016R0679)
Article — Right to erasure ('right to be forgotten')
Applies to: Data subjects (natural persons); controllers (entities determining purpose and means of processing)
Core rule: Data subjects have the right to obtain erasure of their personal data without undue delay; controllers must comply when any of six grounds applies.
Key conditions or exceptions: Grounds: data no longer necessary for original purpose; consent withdrawn and no other legal ground; data subject objects with no overriding legitimate grounds; data unlawfully processed; erasure required by law; data collected from a child online. Exceptions: freedom of expression, legal obligation, public interest in health, archiving/research/statistics, establishment or defence of legal claims.
Procedure: Data subject submits erasure request → controller erases without undue delay → if data was made public, controller must notify other controllers processing it.
Key terms: erasure, right to be forgotten, consent, controller, lawful processing, public interest
Cross-references: Article 6(1)(a), Article 8(1), Article 9(2)(a), Article 21

--- Example 2 ---
INPUT
Act: Data Governance Act (32022R0868) | Article — Notification by data intermediation services providers
Body:
1. Any data intermediation services provider who intends to provide the data intermediation services
referred to in Article 10 shall submit a notification to the competent authority for data intermediation
services. 4. After having submitted a notification [...] the data intermediation services provider may
start the activity [...]. 5. The notification [...] shall entitle the data intermediation services provider
to provide data intermediation services in all Member States. 6. The notification [...] shall include:
(a) the name of the data intermediation services provider; (b) legal status, form, ownership structure;
(c) address of main establishment; (d) a public website [...]; (e) contact details; (f) a description of
the data intermediation service [...]; (g) the estimated date for starting the activity.
8. [...] the competent authority [...] shall, within one week [...], issue a standardised declaration.
11. The competent authority [...] may charge fees [...] proportionate [...]; in the case of SMEs and
start-ups, the competent authority may charge a discounted fee or waive the fee.
12. Data intermediation services providers shall notify [...] any changes [...] within 14 days.

OUTPUT
Act: Data Governance Act (32022R0868)
Article — Notification by data intermediation services providers
Applies to: Data intermediation services providers intending to offer services under Article 10; competent national authorities; the Commission
Core rule: Any provider intending to offer data intermediation services must notify the competent national authority before starting. A single notification grants the right to operate across all EU Member States.
Key conditions or exceptions: Providers not established in the EU must designate a legal representative. Fees must be proportionate and may be reduced or waived for SMEs and start-ups. Changes to notified information must be reported within 14 days.
Procedure: Provider submits notification → may start activity immediately after submission → competent authority issues standardised declaration within 1 week on request → authority notifies Commission → Commission updates public register.
Key terms: notification, data intermediation service, competent authority, standardised declaration, public register, legal representative, SME
Cross-references: Article 10 (types of intermediation services), Article 12 (conditions for provision), Article 13 (competent authorities), Article 33(2)

--- Example 3 ---
INPUT
Act: Data Act (32023R2854) | Article — Obligation to make product data and related service data accessible to the user
Body:
1. Connected products shall be designed and manufactured, and related services shall be designed and
provided, in such a manner that product data and related service data, including the relevant metadata
necessary to interpret and use those data, are, by default, easily, securely, free of charge, in a
comprehensive, structured, commonly used and machine-readable format, and, where relevant and technically
feasible, directly accessible to the user. 2. Before concluding a contract [...] the seller [...] shall
provide at least the following information to the user, in a clear and comprehensible manner:
(a) the type, format and estimated volume of product data; (b) whether the connected product is capable
of generating data continuously and in real-time; (c) whether the connected product is capable of
storing data on-device or on a remote server; (d) how the user may access, retrieve or erase the data.

OUTPUT
Act: Data Act (32023R2854)
Article — Obligation to make product data and related service data accessible to the user
Applies to: Manufacturers of connected products; providers of related services; sellers, rentors, and lessors
Core rule: Connected products must be designed so that product data and metadata are easily, securely, and free of charge accessible to the user by default, in a comprehensive, structured, commonly used, and machine-readable format.
Key conditions or exceptions: Direct accessibility required where relevant and technically feasible. Pre-contractual disclosure of data type, volume, format, storage and access methods is mandatory. Applies to connected products placed on the market after 12 September 2026.
Procedure: Before contract conclusion → seller discloses data type/format/volume/storage/access → user gains access directly from product or by request to data holder.
Key terms: connected product, related service, product data, metadata, machine-readable format, data holder, user, readily available data
Cross-references: Article 2 (definitions), Article 4 (rights and obligations of users and data holders), Article 8(1) GDPR

=== ARTICLE TO SUMMARISE ===
Act: {act_title} ({celex})
Article — {article_title}
Body:
{body}

OUTPUT
"""

CHAPTER_SUMMARY_SYSTEM_V1 = """You are an expert in EU digital legislation (GDPR, AI Act, Data Act, Data Governance Act).
Produce concise retrieval-optimised summaries of legal chapters. These summaries will be
embedded and matched against user questions via semantic similarity search.

Core principle: write for queries, not for readers.
A good chapter summary reads like an answer to "what kinds of question does this chapter answer?" —
not like a table-of-contents entry. Use the plain language a practitioner uses when asking a question;
avoid reproducing the formal register of the legislature.

ANTI-PATTERNS — never do any of the following:
- Restate the chapter title as the Topic sentence (e.g. "This chapter covers requirements applicable to...")
- Use hollow verbs like "establishes", "sets out", "provides for", "lays down"
- List article numbers without explaining what each article does
- Write "Typical questions" as abstract descriptions instead of real user questions

Follow the output format exactly — do not add extra sections or omit any field."""

CHAPTER_SUMMARY_USER_V1 = """Summarise the following legal chapter using the template below.
The input lists the chapter title and the titles of the articles it contains.
If article summaries are provided, use them to write richer Typical questions and Key concepts.

=== OUTPUT TEMPLATE ===
Act: {{regulation_name}} ({{celex}})
Chapter {{number}} — {{title}}
Topic: {{1-2 sentences: what this chapter governs, in plain language that mirrors how a user would describe the problem}}
Applies to: {{entities, roles, and organisations covered — be specific, avoid "relevant parties"}}
Typical questions: {{3-5 verbatim user questions this chapter answers; write them exactly as a user would type them, not as abstract descriptions; cover both broad and narrow questions}}
Key concepts: {{legal terms side-by-side with their plain-language synonyms, comma-separated; include both the formal term and what a non-lawyer would search for}}

=== FEW-SHOT EXAMPLES ===

--- Example 1 ---
INPUT
Act: Data Governance Act (32022R0868)
Chapter I — General provisions
Articles:
- Subject matter and objectives
- Definitions
- Material scope

OUTPUT
Act: Data Governance Act (32022R0868)
Chapter I — General provisions
Topic: Defines the purpose of the Data Governance Act, determines which organisations and data types fall within its scope, and establishes the meaning of its key terms. Draws the boundary between the DGA and related EU instruments such as the GDPR and the Open Data Directive.
Applies to: Public sector bodies, private companies, data holders, data users, data subjects, data intermediation service providers, data altruism organisations, competent authorities, the European Commission.
Typical questions:
What is the Data Governance Act about?
Which organisations does the DGA apply to?
What activities or data types are excluded from the scope of the DGA?
What does 'data intermediation service' mean under the DGA?
What is the difference between a data holder and a data user?
How does the DGA relate to the GDPR?
Key concepts: scope of applicability, subject matter, objectives, definitions, material scope, exclusions, personal data, non-personal data, public sector body, data holder, data user, data subject, data intermediation service, data altruism organisation, re-use

--- Example 2 ---
INPUT
Act: General Data Protection Regulation (32016R0679)
Chapter III — Rights of the data subject
Articles:
- Transparent information, communication and modalities for the exercise of the rights of the data subject
- Information to be provided where personal data are collected from the data subject
- Information to be provided where personal data have not been obtained from the data subject
- Right of access by the data subject
- Right to rectification
- Right to erasure ('right to be forgotten')
- Right to restriction of processing
- Notification obligation regarding rectification or erasure or restriction
- Right to data portability
- Right to object
- Automated individual decision-making, including profiling

OUTPUT
Act: General Data Protection Regulation (32016R0679)
Chapter III — Rights of the data subject
Topic: Grants individuals enforceable rights over their personal data held by controllers: the right to know what is processed and why, to correct or delete it, to obtain a copy in portable format, to restrict processing, and to object to automated decisions and profiling.
Applies to: Data subjects (natural persons whose data is processed), controllers (entities that determine the purpose and means of processing), processors, supervisory authorities.
Typical questions:
Can I ask a company to delete my personal data?
What information must a company give me before collecting my data?
How long does a company have to respond to a data access request?
Can I get a copy of my personal data in a machine-readable format?
Can I opt out of being profiled or targeted by automated decisions?
Is the right to erasure absolute, or are there exceptions?
What happens if a controller shares my data with third parties before I request erasure?
Key concepts: right of access, right to erasure, right to be forgotten, right to rectification, right to data portability, right to object, restriction of processing, automated decision-making, profiling, transparent information, subject access request, response deadline, controller obligations

--- Example 3 ---
INPUT
Act: AI Act (32024R1689)
Chapter III — High-risk AI systems
Articles:
- Classification of AI systems as high-risk
- High-risk AI systems referred to in Annex I
- High-risk AI systems referred to in Annex III
- Risk management system
- Data and data governance
- Technical documentation
- Record-keeping
- Transparency and provision of information to deployers
- Human oversight
- Accuracy, robustness and cybersecurity
- Obligations of providers of high-risk AI systems
- Obligations of product manufacturers
- Authorised representatives of providers established outside the EU
- Obligations of importers
- Obligations of distributors
- Obligations of deployers of high-risk AI systems
- Responsibilities along the AI value chain
- Conformity assessment
- EU declaration of conformity
- CE marking
- Registration

OUTPUT
Act: AI Act (32024R1689)
Chapter III — High-risk AI systems
Topic: Determines which AI systems are classified as high-risk and what technical, organisational, and procedural requirements they must satisfy before and after being placed on the market. Assigns specific compliance obligations to every actor in the AI supply chain — providers, deployers, importers, and distributors.
Applies to: Providers of high-risk AI systems, deployers, importers, distributors, product manufacturers integrating AI, authorised representatives, notified bodies, national market surveillance authorities.
Typical questions:
Which AI systems are considered high-risk under the AI Act?
What technical requirements must a high-risk AI system meet before it can be sold in the EU?
What documentation does a provider need to prepare for a high-risk AI system?
Who is responsible for compliance when a high-risk AI system is embedded in a physical product?
What human oversight measures must be in place for a high-risk AI system?
What obligations does a company deploying a high-risk AI system have?
How does the conformity assessment process work for high-risk AI?
Key concepts: high-risk AI system, Annex I, Annex III, risk management system, technical documentation, conformity assessment, CE marking, EU declaration of conformity, human oversight, data governance, accuracy, robustness, cybersecurity, provider obligations, deployer obligations, AI value chain, notified body, market surveillance

=== CHAPTER TO SUMMARISE ===
Act: {act_title} ({celex})
Chapter {number} — {title}
Articles:
{articles}

OUTPUT
"""

registry.register(PromptVersion(
    name="article_summary_system", version="v1", created=date(2026, 6, 20),
    notes="Initial tracked version. System prompt for retrieval-optimised "
          "article summaries.",
    body=ARTICLE_SUMMARY_SYSTEM_V1, active=True,
))

registry.register(PromptVersion(
    name="article_summary_user", version="v1", created=date(2026, 6, 20),
    notes="Initial tracked version. User prompt with summarisation template "
          "and few-shot examples.",
    body=ARTICLE_SUMMARY_USER_V1, active=True,
))

registry.register(PromptVersion(
    name="chapter_summary_system", version="v1", created=date(2026, 6, 25),
    notes="Initial version. System prompt for retrieval-optimised chapter summaries. "
          "Emphasises query-friendly language and bans hollow legislative phrasing.",
    body=CHAPTER_SUMMARY_SYSTEM_V1, active=True,
))

registry.register(PromptVersion(
    name="chapter_summary_user", version="v1", created=date(2026, 6, 25),
    notes="Initial version. User prompt with 3 few-shot examples covering a generic "
          "(Ch. I General provisions), a rights-based (GDPR Ch. III), and a complex "
          "domain-specific chapter (AI Act Ch. III High-risk AI).",
    body=CHAPTER_SUMMARY_USER_V1, active=True,
))
