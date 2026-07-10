FROM gcr.io/oss-fuzz-base/base-builder-rust@sha256:a960440a8543df2411ca489eff995366ce8041b086141cc5092664d793f20e9e

RUN groupadd --gid 1001 fuzzbuilder \
    && useradd --uid 1001 --gid fuzzbuilder --create-home --shell /bin/bash fuzzbuilder \
    && touch /usr/lib/libFuzzingEngine.a \
    && chown fuzzbuilder:fuzzbuilder /usr/lib/libFuzzingEngine.a \
    && chown -R fuzzbuilder:fuzzbuilder /rust /out /work /src

COPY --chown=fuzzbuilder:fuzzbuilder . $SRC/fast-mlsirm
WORKDIR $SRC/fast-mlsirm
USER fuzzbuilder
