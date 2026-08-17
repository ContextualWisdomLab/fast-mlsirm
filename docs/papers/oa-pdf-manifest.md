# OA PDF manifest

This manifest records the original PDFs intentionally preserved in this
repository. Hashes are SHA-256 of the committed bytes. Zotero attachment keys
are local provenance only; they are not credentials and are not required by CI.

| File | Source | Reuse basis | Zotero attachment | Bytes | SHA-256 |
| --- | --- | --- | --- | ---: | --- |
| [`li-2025-evaluating-scoring-bias-llm-as-a-judge.pdf`](papers/li-2025-evaluating-scoring-bias-llm-as-a-judge.pdf) | [arXiv:2506.22316](https://arxiv.org/abs/2506.22316), [PDF](https://arxiv.org/pdf/2506.22316) | arXiv record states CC BY 4.0 | `TVZMTEB8` | 786,966 | `1ea7b239ff1341189bd3927bbab63b43b190289437156d89e0e251aca146b744` |
| [`pezeshkpour-hruschka-2024-option-order-sensitivity.pdf`](papers/pezeshkpour-hruschka-2024-option-order-sensitivity.pdf) | [ACL Anthology page](https://aclanthology.org/2024.findings-naacl.130/), [PDF](https://aclanthology.org/2024.findings-naacl.130.pdf) | ACL Anthology states materials published in or after 2016 are CC BY 4.0 | `S5KQCN97` | 439,433 | `4d0fecf2b1da9544112c286a699f8873b0c7a4ce2f38f400bf09db5f10659624` |
| [`sharma-2023-sycophancy.pdf`](papers/sharma-2023-sycophancy.pdf) | [arXiv:2310.13548](https://arxiv.org/abs/2310.13548), [PDF](https://arxiv.org/pdf/2310.13548) | arXiv record states CC BY 4.0 | `47VH4PC7` | 1,383,108 | `ee764bd30119f2146f2e130a099d6d313fca6c70ab07b17b7fdbde456d96be36` |

## Citation-only records

- Jones and Loe (2013), [SAGE Open page](https://journals.sagepub.com/doi/10.1177/2158244013489691): OpenAlex/Unpaywall identify the published version as gold OA with CC BY metadata, and [ResearchGate lists an author-uploaded CC BY 3.0 copy](https://www.researchgate.net/publication/258187383_Optimal_Number_of_Questionnaire_Response_Categories_More_May_Not_Be_Better). An external official PDF fetch confirms the 10-page source, but the reproducible local downloader receives HTTP 403 HTML. Zotero Local API item `CWY355RP` confirms open access and has `numChildren=0`. Zotero 9.0.6 exposes read-only Local API behavior here, so no substitute or regenerated PDF is counted as the original; add the original binary only after an authorized Zotero/Web API or GUI retrieval.
- Zheng et al. (2024), [ICLR page](https://proceedings.iclr.cc/paper_files/paper/2024/hash/54dd9e0cff6d9214e20d97eb2a3bae49-Abstract-Conference.html): the local arXiv record grants a non-exclusive distribution license to arXiv, and the ICLR policy describes a license granted to ICLR; neither is an explicit third-party repository redistribution license.
- Iannario, Monti, and Scalera (2022), [publisher page](https://www.degruyterbrill.com/document/doi/10.1515/ijb-2021-0013/html): publisher and Crossref/OpenAlex metadata identify CC BY 4.0, and Zotero item `MYPNHHWJ` confirms that license. An external official PDF fetch confirms the 19-page source and CC BY 4.0 notice, but the reproducible local redirect returns CloudFront WAF HTTP 202 with zero bytes and the local item has no child attachment. The original binary remains an OA retrieval task rather than a vendored file.
- Cao et al. (2026), [AAAI landing page](https://ojs.aaai.org/index.php/AAAI/article/view/40262) and [official PDF](https://ojs.aaai.org/index.php/AAAI/article/view/40262/44223): the PDF states all rights reserved, so it is citation-only despite being publicly readable.
- Samejima (1969), [Psychometric Society reproduction](https://www.psychometricsociety.org/sites/main/files/file-attachments/mn17.pdf): publicly accessible and preserved in Zotero for local research, but no redistribution license is asserted here.

The Cao citation was added through the local Zotero Connector API on
2026-08-14 as item `393S5NXZ`; the official PDF was intentionally not copied.

## Interpretation boundary

These papers motivate paired option-order, category-count, rubric-order, score-ID,
and sycophancy probes. They do not prove a universal positive effect of larger
`K`. The repository therefore retains K, method, prompt/order variant, parse
status, provider readiness, and human/gold anchor fields as calibration evidence;
it does not convert any of these papers into a keyword-matching judge.
