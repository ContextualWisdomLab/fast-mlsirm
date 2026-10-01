# G1 모수 대조용 합성 입력

`sparse_p2_s4.json`은 기존 `_continuous_six_latent_fixture`로 만든 64명×16문항 P2/S4 합성 입력이다. 참가자 자료가 아니다. CPython 3.12.14, seed=20260930, latent mean=0·SD=1을 사용했다. 응답의 little-endian int64 SHA-256은 `d829e8fd51d22eb73da4f25f73cbe828f0386ab0c23e1c0e53f5e8d4ac589ee8`이다.

Φ=I, q7/7, n_starts=1, tol=1e-6에서 CPU/GPU 문항 모수 차이 0.321을 재현한다. 낮은 node count·작은 표본의 실행 회귀 입력이며 연구 설정 수용이나 모수 회복 증거가 아니다. 수렴·likelihood 근접만으로 모수 동등성을 통과시키지 않기 위한 입력이다. 제약이나 prior를 추가해 차이를 가리지 않는다.

생성 근거: Cai, L. (2010). A two-tier full-information item factor analysis model with applications. *Psychometrika, 75*(4), 581–612. https://doi.org/10.1007/s11336-010-9178-0 — 기존 테스트 생성기가 사용하는 pp.587–589, equations 4–7/9/11–12; 적합 대조는 직접 읽은 pp.608–609 Appendices A/B를 따른다.
