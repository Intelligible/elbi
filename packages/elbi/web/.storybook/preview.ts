import type { Preview } from "@storybook/react-vite"

// Load the design tokens so stories render with the real theme, not Storybook defaults.
import "../src/index.css"

const preview: Preview = {
  parameters: {
    layout: "centered",
    controls: { matchers: { color: /(background|color)$/i } },
  },
}

export default preview
