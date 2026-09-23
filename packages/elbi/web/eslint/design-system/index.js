import allowedShadow from "./rules/allowed-shadow.js"
import knownColorToken from "./rules/known-color-token.js"
import noArbitraryValue from "./rules/no-arbitrary-value.js"
import noColorLiteral from "./rules/no-color-literal.js"
import noLegacyCss from "./rules/no-legacy-css.js"
import noRawElement from "./rules/no-raw-element.js"

export default {
  meta: { name: "design-system" },
  rules: {
    "no-raw-element": noRawElement,
    "no-arbitrary-value": noArbitraryValue,
    "no-color-literal": noColorLiteral,
    "no-legacy-css": noLegacyCss,
    "allowed-shadow": allowedShadow,
    "known-color-token": knownColorToken,
  },
}
