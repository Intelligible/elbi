/** The title and one-line explanation above a settings section.
 *
 * Shared so a section extracted into its own file heads itself exactly like one that
 * still lives in `SettingsPage`; the alternative was each extracted file rolling its
 * own, which is how the type scale drifts apart a heading at a time.
 */
export function SectionHeader({ title, hint }: { title: string; hint: string }) {
  return (
    <div className="mb-5">
      <h2 className="text-base font-semibold">{title}</h2>
      <p className="mt-1 text-sm text-text-tertiary">{hint}</p>
    </div>
  )
}
