export type Theme = { colors: Set<string>; textSizes: Set<string> }
export function readTheme(cssPath: string): Theme
