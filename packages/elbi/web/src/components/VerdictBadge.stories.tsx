import type { Meta, StoryObj } from "@storybook/react-vite"

import { VerdictBadge } from "./VerdictBadge"

// The oracle's verdict vocabulary, rendered from one source of truth. Only a
// certified-sound run wears the verified tint.
const meta: Meta<typeof VerdictBadge> = {
  title: "Verdict/VerdictBadge",
  component: VerdictBadge,
  args: { verdict: "sound" },
}
export default meta

type Story = StoryObj<typeof VerdictBadge>

export const Sound: Story = { args: { verdict: "sound" } }
export const Unsound: Story = { args: { verdict: "unsound" } }
export const Inconclusive: Story = { args: { verdict: "inconclusive" } }
export const Computed: Story = { args: { verdict: "computed" } }
export const Unverified: Story = { args: { verdict: "unverified" } }
