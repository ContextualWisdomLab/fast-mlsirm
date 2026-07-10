FROM gcr.io/oss-fuzz-base/base-builder-rust@sha256:a960440a8543df2411ca489eff995366ce8041b086141cc5092664d793f20e9e

RUN useradd --create-home --shell /bin/bash fuzzbuilder \
    && chown -R fuzzbuilder:fuzzbuilder /rust

COPY --chown=fuzzbuilder:fuzzbuilder . $SRC/fast-mlsirm
WORKDIR $SRC/fast-mlsirm
USER fuzzbuilder
