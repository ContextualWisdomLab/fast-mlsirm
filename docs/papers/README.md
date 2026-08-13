# Reference Papers

Curated references that ground the estimation core and the contextual-
orchestrator-backed LLM-judge boundary. Citation-only references stay in this
directory; `papers/` contains only original PDFs whose source explicitly allows
redistribution (or whose repository/archive terms are clear). Zotero presence
or public downloadability alone is not treated as a redistribution license.

The retained files and their source/license/hash checks are recorded in
[`oa-pdf-manifest.md`](oa-pdf-manifest.md). No PDF is fetched through a
provider credential, and no paper is used as a keyword-matching judgment rule.

## Wu et al. 2021

Wu, M., Davis, R. L., Domingue, B. W., Piech, C., & Goodman, N. (2021).
*Modeling Item Response Theory with Stochastic Variational Inference.*
arXiv:2108.11579. https://arxiv.org/abs/2108.11579

- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0),
  https://creativecommons.org/licenses/by/4.0/.
- **Why it is referenced:** it develops a *fast, scalable* estimator for item
  response theory by mapping the likelihood and its gradient onto
  data-parallel, accelerator-friendly computation (amortized/stochastic
  variational inference). That is exactly the numerical shape this project
  accelerates: the penalized negative log-likelihood and gradient hot path of
  the MLSIRM/MLS2PLM family, now offloadable to the GPU via the wgpu GPGPU
  kernels in `crates/mlsirm-core/src/gpu.rs`. The paper is the design reference
  for keeping GPU-accelerated IRT estimation numerically faithful to the CPU
  objective.

## LLM-judge and option/category-bias references

Li, Q., Dou, S., Shao, K., Chen, C., & Hu, H. (2025). *Evaluating scoring bias
in LLM-as-a-Judge*. arXiv:2506.22316. <https://arxiv.org/abs/2506.22316>.
CC BY 4.0. The original PDF is preserved as
[`papers/li-2025-evaluating-scoring-bias-llm-as-a-judge.pdf`](papers/li-2025-evaluating-scoring-bias-llm-as-a-judge.pdf).

Pezeshkpour, P., & Hruschka, E. (2024). Large language models sensitivity to
the order of options in multiple-choice questions. *Findings of ACL: NAACL
2024*, 2006–2017. <https://doi.org/10.18653/v1/2024.findings-naacl.130>.
ACL Anthology materials published in or after 2016 are CC BY 4.0. The original
PDF is preserved as
[`papers/pezeshkpour-hruschka-2024-option-order-sensitivity.pdf`](papers/pezeshkpour-hruschka-2024-option-order-sensitivity.pdf).

Sharma, M., Tong, M., Korbak, T., et al. (2023). *Towards understanding
sycophancy in language models*. arXiv:2310.13548.
<https://arxiv.org/abs/2310.13548>. CC BY 4.0. The original PDF is preserved as
[`papers/sharma-2023-sycophancy.pdf`](papers/sharma-2023-sycophancy.pdf).

Zheng, C., Zhou, H., Meng, F., Zhou, J., & Huang, M. (2024). *Large language
models are not robust multiple choice selectors*. ICLR 2024.
<https://proceedings.iclr.cc/paper_files/paper/2024/hash/54dd9e0cff6d9214e20d97eb2a3bae49-Abstract-Conference.html>.
This remains citation-only: the Zotero/arXiv record exposes arXiv's
non-exclusive license to distribute to arXiv, while the ICLR policy grants
distribution rights to ICLR; neither is an explicit third-party repository
redistribution license for this project.

Samejima, F. (1969). *Estimation of latent ability using a response pattern of
graded scores*. *Psychometrika, 34*(Suppl. 1), 1–97.
<https://doi.org/10.1007/BF03372160>. The Psychometric Society reproduction is
linked from ADR-0014 but remains citation-only because its accessible PDF does
not state a redistribution license.
