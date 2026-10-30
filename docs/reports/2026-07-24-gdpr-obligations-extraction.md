# GDPR obligation extraction report

**Act:** Regulation (EU) 2016/679 (GDPR) — CELEX 32016R0679
**Date:** 2026-07-24
**Model:** gpt-5-mini (extraction), text-embedding-3-small (anchoring)
**Run:** full act, 8-way concurrency, ~19 minutes, no failed calls.

## Summary

| Metric | Value |
|---|---|
| Candidate passages (detection) | 335 / 414 paragraphs |
| Obligations extracted | 471 |
| Obligations with no resolved addressee | 124 |
| Actor nodes (11 curated + 43 promoted) | 54 |
| Graph writes | 525 nodes, 1299 edges |

## Obligation type (paper Table 3 analogue)

| k | n |
|---|---|
| ACTION | 394 |
| BEING | 77 |

## Modality

| k | n |
|---|---|
| OBLIGATION | 417 |
| PROHIBITION | 54 |

## Trust: weakest extraction method per obligation

How reliable each obligation's weakest element is. STATED is drawn from the text; BACKGROUND
was inferred by the model with no textual basis.

| k | n |
|---|---|
| STATED | 321 |
| CONTEXT | 116 |
| BACKGROUND | 24 |
| CITATION | 10 |

## Addressees (paper Table 4 analogue)

Who bears the duties, by resolved actor.

| actor | n |
|---|---|
| supervisory_authority_concerned | 109 |
| supervisory_authority | 109 |
| controller | 90 |
| the_board | 47 |
| processor | 41 |
| the_commission | 39 |
| each_member_state | 17 |
| member_states | 11 |
| the_data_protection_officer | 9 |
| the_chair_of_the_board | 6 |
| as_the_case_may_be | 5 |
| where_applicable | 4 |
| the_chair | 4 |
| the_member_states | 3 |
| such_controllers | 3 |
| a_body_as_referred_to_in_paragraph_1_of_this_article | 3 |
| the_secretariat | 3 |
| data_subject | 2 |
| the_controller_s_representative | 2 |
| each_controller | 2 |
| other_bodies_representing_categories_of_controllers | 2 |
| more_controllers_joint_controllers | 1 |
| the_joint_controllers | 1 |
| the_initial_processor | 1 |
| any_person_acting_under_the_authority_of_the_controller | 1 |
| the_processor_s_representative_where_applicable | 1 |
| the_processor_s_representative | 1 |
| representative | 1 |
| the_processor_who_has_access_to_personal_data | 1 |
| any_natural_person_acting_under_the_authority_of_the_controller | 1 |
| processor_which_submits_its_processing_to_the_certification_mechanism | 1 |
| the_member_state_of_the_seconding_supervisory_authority_whose_staff_has_caused_damage_to_any_person_in_the_territory_of_another_member_state | 1 |
| controllers_involved_in_processing_data_controllers | 1 |
| data_controllers | 1 |

## Beneficiaries (paper Table 6 analogue, top 25)

| actor | n |
|---|---|
| data_subject | 79 |
| supervisory_authority | 45 |
| supervisory_authority_concerned | 45 |
| the_commission | 25 |
| controller | 16 |
| the_board | 13 |
| processor | 10 |
| the_public | 8 |
| natural_persons | 5 |
| the_complainant | 4 |
| the_european_parliament | 3 |
| other_authorities_as_designated_by_member_state_law | 3 |
| the_government | 3 |
| the_national_parliament | 3 |
| the_data_protection_officer | 3 |
| the_council | 3 |
| single_establishment_of_the_controller | 2 |
| as_the_case_may_be | 2 |
| where_applicable | 1 |
| the_highest_management_level_of_the_controller | 1 |
| each_recipient_to_whom_the_personal_data_have_been_disclosed | 1 |
| group_of_undertakings | 1 |
| processor_on_the_territory_of_its_member_state | 1 |
| medium_sized_enterprises | 1 |
| controllers_which_are_processing_the_personal_data | 1 |

## Predicates (paper Table 5 analogue, top 25)

| pred | n |
|---|---|
| shall inform | 21 |
| shall provide | 16 |
| shall notify | 14 |
| shall be adopted | 12 |
| shall communicate | 11 |
| shall submit | 9 |
| shall ensure | 6 |
| shall adopt | 5 |
| shall publish | 4 |
| shall issue guidelines, recommendations and best practices | 4 |
| shall notify to the Commission | 4 |
| shall cooperate | 4 |
| shall consult | 3 |
| shall at least describe | 3 |
| shall be provided | 3 |
| shall take into account | 3 |
| shall be issued | 3 |
| shall designate | 3 |
| shall collate | 3 |
| shall include | 3 |
| shall be made public | 3 |
| shall, in particular, take account of | 3 |
| shall encourage | 2 |
| shall establish | 2 |
| shall document | 2 |

## Promoted actors (43)

Actors the Definitions article (Article 4) does not carry, promoted from the extraction's
addressee strings. Committed to `actors.yaml`, pending review.

### With a parent (qualified forms, IS_A working)

- `a_data_subject_who_has_obtained_restriction_of_processing_pursuant_to_paragraph_1` → ['data_subject']
- `any_natural_person_acting_under_the_authority_of_the_controller` → ['controller']
- `any_person_acting_under_the_authority_of_the_controller` → ['controller']
- `controllers_involved_in_processing_data_controllers` → ['data_controllers']
- `controllers_which_are_processing_the_personal_data` → ['controller']
- `data_controllers` → ['controller']
- `each_controller` → ['controller']
- `each_recipient_to_whom_the_personal_data_have_been_disclosed` → ['recipient']
- `medium_sized_enterprises` → ['enterprise']
- `more_controllers_joint_controllers` → ['controller']
- `other_bodies_representing_categories_of_controllers` → ['controller']
- `processor_on_the_territory_of_its_member_state` → ['processor']
- `processor_which_submits_its_processing_to_the_certification_mechanism` → ['processor']
- `single_establishment_of_the_controller` → ['controller']
- `such_controllers` → ['controller']
- `the_controller_s_representative` → ['representative']
- `the_highest_management_level_of_the_controller` → ['controller']
- `the_initial_processor` → ['processor']
- `the_joint_controllers` → ['controller']
- `the_member_state_of_the_seconding_supervisory_authority_whose_staff_has_caused_damage_to_any_person_in_the_territory_of_another_member_state` → ['supervisory_authority']
- `the_processor_s_representative` → ['representative']
- `the_processor_s_representative_where_applicable` → ['the_processor_s_representative']
- `the_processor_who_has_access_to_personal_data` → ['processor']

### Standalone (institutional actors + noise)

- `a_body_as_referred_to_in_paragraph_1_of_this_article`
- `as_the_case_may_be`
- `each_member_state`
- `member_states`
- `natural_persons`
- `other_authorities_as_designated_by_member_state_law`
- `the_board`
- `the_chair`
- `the_chair_of_the_board`
- `the_commission`
- `the_complainant`
- `the_council`
- `the_data_protection_officer`
- `the_european_parliament`
- `the_government`
- `the_member_states`
- `the_national_parliament`
- `the_public`
- `the_secretariat`
- `where_applicable`

## Unmapped addressee strings

82 strings matched no actor and recurred below the promotion floor (3), or matched nothing.
Not persisted in the graph; this is the sample logged at ingest time:

| string | count |
|---|---|
| others | 2 |
| associations | 2 |
| the certification bodies referred to in Article 43 | 2 |
| certification bodies which have an appropriate level of expertise in relation to data protection | 2 |
| The certification bodies referred to in paragraph 1 | 2 |
| that Member State | 2 |
| The member | 2 |
| Member | 2 |
| the main establishment | 2 |
| each other | 2 |

## Known quality issues (from this run)

1. **Noise promoted by frequency.** `as_the_case_may_be`, `where_applicable`, `the_public`
   are not parties; the frequency floor alone does not tell a role from a recurring phrase.
2. **Fragmentation.** `each_member_state`, `member_states`, `the_member_states` are three
   actors that should be one. Containment does not normalise morphological variants.
3. **Verbose clauses.** The model sometimes extracts a whole clause as an addressee, promoted
   verbatim (e.g. `a_data_subject_who_has_obtained_restriction_of_processing_pursuant_to_paragraph_1`).
   The IS_A is correct; the id is a sentence, not a role.

The obligations themselves are correctly linked; the noise is in the vocabulary, which is a
committed, reviewable artefact by design.
