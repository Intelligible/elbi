import babelParser from "@babel/eslint-parser"

// Babel, not typescript-eslint: the repo is on TypeScript 7, which typescript-eslint cannot load.
export const languageOptions = {
  parser: babelParser,
  parserOptions: {
    requireConfigFile: false,
    babelOptions: {
      babelrc: false,
      configFile: false,
      parserOpts: { plugins: ["jsx", "typescript"] },
    },
  },
}
