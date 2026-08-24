# UI/UX and Storybook evidence boundary

Status: **downstream acceptance contract**

`fast-mlsirm` has no web frontend. This record prevents a downstream host from
calling a static screenshot or a passing numerical test UI evidence. When a
consumer adds a web surface, it owns the Figma file, design tokens, Storybook
stories, browser test runner, and release evidence.

## Evidence rule

One Storybook story represents one observable scene. The story declares its
initial props and context. Its `play` function performs a realistic user event
such as keyboard navigation, pointer/touch input, typing, submission, retry,
cancel, route change, or export. Assertions use the accessible DOM and public
result surfaces: role/name, focus, visible status, callback arguments, URL,
exact-value table, or serialized output. Implementation-only selectors do not
replace an observable assertion.

Each component inventory must cover these dimensions and edge families:

1. accessibility;
2. touch and interaction;
3. performance;
4. style selection and design-token regression;
5. layout and responsive behavior;
6. typography and color;
7. animation and reduced motion;
8. forms and feedback;
9. navigation patterns; and
10. charts and data, including an exact-value alternative.

The minimum state set is: empty, populated, long-content, loading, disabled,
invalid, server-error, retry, success, cancellation, no-data, and permission
denied where the component supports the state. The host records viewport,
locale, color scheme, reduced-motion preference, data fixture identity, and
test result with the release artifact.

## Design identity

For the existing buyer-review packet, the Figma file ID is
`qD34PfMH8Kr41tFdqLCkem`. Its ADR binding is tracked by
[PR #1130](https://github.com/ContextualWisdomLab/fast-mlsirm/pull/1130);
until that PR merges, the identity is active-PR evidence rather than protected
main truth. Figma Code Connect remains disabled by the packet contract.

## Authority and research traceability

WCAG 2.2 is the W3C Recommendation baseline for web accessibility. Storybook's
official testing guidance defines stories as UI test cases and recommends
`play` functions for user interaction and assertions. These sources govern the
UI evidence shape; they do not certify a downstream product or this numerical
library.

### APA 7th references

World Wide Web Consortium. (2024). *Web Content Accessibility Guidelines
(WCAG) 2.2*. W3C Recommendation. https://www.w3.org/TR/WCAG22/

Storybook. (n.d.). *Interaction tests*. Storybook documentation.
https://storybook.js.org/docs/writing-tests/interaction-testing

Storybook. (n.d.). *How to test UIs with Storybook*. Storybook documentation.
https://storybook.js.org/docs/writing-tests
