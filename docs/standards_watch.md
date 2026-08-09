# Standards and research watch

## Status and use

This registry is part of the proposed canonical architecture baseline. It separates current published sources used as governing references from drafts, revisions, and emerging research that are monitored but not treated as normative. Catalog status and edition numbers must be revalidated from the issuing body's official publication record before every release that claims alignment.

`fast-mlsirm` does not claim certification or conformance merely because a standard is cited. Applicability, control implementation, independent assessment, operating evidence, and downstream responsibilities remain separate questions.

## Governing published references

| Area | Published reference used by the architecture baseline | Repository use |
|---|---|---|
| Requirements engineering | ISO/IEC/IEEE 29148:2018 | PRD/TRD quality, requirement attributes, traceability, verification and change control |
| Architecture description | ISO/IEC/IEEE 42010:2022 | stakeholders, concerns, viewpoints, views, correspondence and decision records |
| Product quality | ISO/IEC 25010:2023 | functional suitability, performance efficiency, compatibility, interaction capability, reliability, security, maintainability, flexibility and safety quality evidence |
| AI management system | ISO/IEC 42001:2023 | lifecycle governance, roles, change control, documentation and operating evidence; no certification claim |
| AI impact assessment | ISO/IEC 42005:2025 | downstream use-context impact assessment and recorded human/governance decisions |
| AI risk management | ISO/IEC 23894:2023 | risk identification, analysis, treatment, monitoring and communication |
| AI risk framework | NIST AI RMF 1.0 | Govern, Map, Measure and Manage control framing |
| Generative AI profile | NIST AI 600-1 | model/provider risks, content provenance, human oversight, evaluation and incident considerations |
| Testing validity | *Standards for Educational and Psychological Testing* (2014) | score interpretation, fairness, reliability/precision, validation and use boundaries |
| Web accessibility | WCAG 2.2 | standalone HTML report semantics, focus, contrast, reflow, target size, status messages and non-hover exact-value channels |

## Normative-versus-watch policy

1. A published, applicable edition may govern a requirement or ADR after its exact edition and official source are recorded.
2. A committee draft, working draft, public consultation, amendment proposal, revision project, or announced future edition is a **watch item**, not a normative requirement.
3. When a new edition is published, maintainers perform a delta assessment before changing repository requirements. The assessment records superseded clauses, migration impact, implementation evidence, release impact, and downstream ownership.
4. A citation does not establish implementation, certification, legal compliance, or suitability for a regulated decision.
5. Scientific method claims use primary peer-reviewed papers where available. Package documentation or legacy software can be a numerical comparison source but does not replace primary methodological validation.

## Active watch items

The following topics are monitored because revisions or new evidence may change future requirements. Their exact project identifiers and publication state must be checked against official sources at review time.

- revisions to requirements-engineering and architecture-description standards;
- updates to the ISO/IEC 25000 SQuaRE family affecting measurement or quality models;
- implementation guidance and conformity-assessment practice for ISO/IEC 42001 and ISO/IEC 42005;
- NIST AI RMF and Generative AI Profile revisions, implementation resources, and evaluation guidance;
- revision of the *Standards for Educational and Psychological Testing*;
- later W3C accessibility recommendations and techniques beyond WCAG 2.2;
- current primary evidence on LLM-as-a-Judge calibration, dynamic rubrics, automatic item generation, test-time scaling, multi-agent verification, and correlated evaluator error;
- current primary evidence for multilevel, multiple-membership, longitudinal, testlet, bifactor, two-tier, latent-space, factor-retention and model-selection methods.

## Release review checklist

Before a release or buyer evidence packet uses a standards claim:

- verify official publication status, edition, title, and issuing body;
- identify the exact requirement or concern affected;
- link the requirement to an ADR, implementation, test, and evidence artifact;
- distinguish core-library obligations from Psychometrics Commons or another downstream host;
- record gaps, compensating controls, migration needs, and rejected applicability;
- remove language that implies certification, conformity, safety, fairness, validity, or legal compliance without independent evidence;
- preserve the published edition as governing until an adopted replacement is approved through an ADR.

## APA 7 reference record

American Educational Research Association, American Psychological Association, & National Council on Measurement in Education. (2014). *Standards for educational and psychological testing*. American Educational Research Association.

International Organization for Standardization & International Electrotechnical Commission. (2023a). *ISO/IEC 23894:2023 Information technology—Artificial intelligence—Guidance on risk management*.

International Organization for Standardization & International Electrotechnical Commission. (2023b). *ISO/IEC 25010:2023 Systems and software engineering—Systems and software quality requirements and evaluation (SQuaRE)—Product quality model*.

International Organization for Standardization & International Electrotechnical Commission. (2023c). *ISO/IEC 42001:2023 Information technology—Artificial intelligence—Management system*.

International Organization for Standardization & International Electrotechnical Commission. (2025). *ISO/IEC 42005:2025 Information technology—Artificial intelligence—AI system impact assessment*.

International Organization for Standardization, International Electrotechnical Commission, & Institute of Electrical and Electronics Engineers. (2018). *ISO/IEC/IEEE 29148:2018 Systems and software engineering—Life cycle processes—Requirements engineering*.

International Organization for Standardization, International Electrotechnical Commission, & Institute of Electrical and Electronics Engineers. (2022). *ISO/IEC/IEEE 42010:2022 Software, systems and enterprise—Architecture description*.

National Institute of Standards and Technology. (2023). *Artificial intelligence risk management framework (AI RMF 1.0)* (NIST AI 100-1). https://doi.org/10.6028/NIST.AI.100-1

National Institute of Standards and Technology. (2024). *Artificial intelligence risk management framework: Generative artificial intelligence profile* (NIST AI 600-1). https://doi.org/10.6028/NIST.AI.600-1

World Wide Web Consortium. (2023). *Web content accessibility guidelines (WCAG) 2.2*. https://www.w3.org/TR/WCAG22/
