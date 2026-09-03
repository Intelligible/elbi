import { Check, Pencil, Plus, Star, Trash2 } from "lucide-react"
import { useCallback, useEffect, useState } from "react"
import { Button } from "@/components/ui/button"
import { useFeedback } from "@/components/ui/feedback"
import {
  deleteLlmProfile,
  getLlmProfiles,
  type LlmProfile,
  saveLlmProfile,
  setDefaultProfile,
  setTitleProfile,
} from "@/lib/chat"

const inputClass =
  "w-full rounded-lg border border-border bg-background px-3 py-2 text-sm outline-none focus:ring-1 focus:ring-ring"

// The model-profile manager as a settings-page panel. Registering several profiles (each a
// model + API key + base URL) lets the app run on any provider (OpenAI, Claude, a local model)
// and a conversation pick which one it uses, all without a restart. Also sets the cheap "title
// model" used to name chats.
export function LlmProfilesManager() {
  const [profiles, setProfiles] = useState<LlmProfile[]>([])
  const [defaultName, setDefaultName] = useState("")
  const [titleName, setTitleName] = useState("")
  const [editing, setEditing] = useState<LlmProfile | "new" | null>(null)
  const fb = useFeedback()

  const refresh = useCallback(async () => {
    const p = await getLlmProfiles()
    setProfiles(p.profiles)
    setDefaultName(p.default)
    setTitleName(p.titleProfile)
  }, [])

  useEffect(() => {
    void refresh()
  }, [refresh])

  if (editing !== null) {
    return (
      <ProfileEditor
        profile={editing === "new" ? null : editing}
        existingNames={profiles.map((p) => p.name)}
        onDone={async () => {
          await refresh()
          setEditing(null)
        }}
        onCancel={() => setEditing(null)}
      />
    )
  }

  return (
    <div className="space-y-3">
      <div className="space-y-1.5">
        {profiles.length === 0 ? (
          <div className="rounded-xl border border-dashed border-border px-4 py-8 text-center text-sm text-muted-foreground">
            No profiles yet. Add one to run on OpenAI, Claude, a local model, and more.
          </div>
        ) : (
          profiles.map((p) => (
            <ProfileRow
              key={p.name}
              profile={p}
              isDefault={p.name === defaultName}
              onEdit={() => setEditing(p)}
              onDelete={async () => {
                if (
                  !(await fb.confirm({
                    title: `Delete profile "${p.name}"?`,
                    body: "Conversations using it fall back to the default model.",
                    danger: true,
                  }))
                )
                  return
                if (await deleteLlmProfile(p.name)) fb.toast("ok", `Deleted "${p.name}"`)
                else fb.toast("error", `Could not delete "${p.name}"`)
                await refresh()
              }}
              onSetDefault={async () => {
                await setDefaultProfile(p.name)
                await refresh()
              }}
            />
          ))
        )}
      </div>
      {profiles.length > 0 && (
        <label className="flex items-center justify-between gap-2 border-t border-border pt-3 text-xs text-muted-foreground">
          <span>
            Title model
            <span className="ml-1">(a cheap model to name chats)</span>
          </span>
          <select
            value={titleName}
            onChange={async (e) => {
              await setTitleProfile(e.target.value)
              await refresh()
            }}
            className="rounded-lg border border-border bg-background px-2 py-1 text-xs disabled:opacity-60"
          >
            <option value="">Default</option>
            {profiles.map((p) => (
              <option key={p.name} value={p.name}>
                {p.name}
              </option>
            ))}
          </select>
        </label>
      )}
      <Button variant="outline" size="sm" onClick={() => setEditing("new")}>
        <Plus className="h-4 w-4" /> Add profile
      </Button>
    </div>
  )
}

function ProfileRow({
  profile,
  isDefault,
  onEdit,
  onDelete,
  onSetDefault,
}: {
  profile: LlmProfile
  isDefault: boolean
  onEdit: () => void
  onDelete: () => void
  onSetDefault: () => void
}) {
  return (
    <div className="flex items-center gap-2 rounded-lg border border-border px-3 py-2">
      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-1.5">
          <span className="truncate text-sm font-medium">{profile.name}</span>
          {isDefault && (
            <span className="inline-flex items-center gap-1 rounded-full bg-foreground/10 px-1.5 text-[10px] text-muted-foreground">
              <Check className="h-3 w-3" /> default
            </span>
          )}
        </div>
        <span className="truncate text-xs text-muted-foreground">
          {profile.model || "(no model set)"}
          {profile.apiKeySet ? " · key set" : ""}
        </span>
      </div>
      {!isDefault && (
        <Button variant="ghost" size="icon-xs" aria-label="Set as default" onClick={onSetDefault}>
          <Star className="h-3.5 w-3.5" />
        </Button>
      )}
      <Button variant="ghost" size="icon-xs" aria-label="Edit profile" onClick={onEdit}>
        <Pencil className="h-3.5 w-3.5" />
      </Button>
      <Button
        variant="ghost"
        size="icon-xs"
        aria-label="Delete profile"
        className="text-destructive"
        onClick={onDelete}
      >
        <Trash2 className="h-3.5 w-3.5" />
      </Button>
    </div>
  )
}

function ProfileEditor({
  profile,
  existingNames,
  onDone,
  onCancel,
}: {
  profile: LlmProfile | null
  existingNames: string[]
  onDone: () => void
  onCancel: () => void
}) {
  const isNew = profile === null
  const [name, setName] = useState(profile?.name ?? "")
  const [model, setModel] = useState(profile?.model ?? "")
  const [baseUrl, setBaseUrl] = useState(profile?.baseUrl ?? "")
  const [apiKey, setApiKey] = useState("")
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState("")

  const duplicate = isNew && existingNames.includes(name.trim())
  const canSave = name.trim().length > 0 && !duplicate && !saving
  // A model string with no "/" is sent to Anthropic as
  // "anthropic/<value>" rather than rejected. That is the intended shorthand for a bare
  // Anthropic id (e.g. "claude-sonnet-5"), but the same rule silently misroutes any other
  // provider's bare model name. Surfaced here instead of only in that error.
  const impliesAnthropic = model.trim() !== "" && !model.includes("/")

  const save = async () => {
    setSaving(true)
    setError("")
    const patch: { model: string; base_url: string; api_key?: string } = {
      model: model.trim(),
      base_url: baseUrl.trim(),
    }
    if (apiKey.trim()) patch.api_key = apiKey.trim()
    try {
      await saveLlmProfile(name.trim(), patch)
      onDone()
    } catch (e) {
      // Surface the server's reason (e.g. an invalid name) instead of failing silently.
      setError(e instanceof Error ? e.message : "Could not save this profile.")
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="space-y-3">
      <label className="block space-y-1">
        <span className="text-xs font-medium">Name</span>
        <input
          value={name}
          onChange={(e) => setName(e.target.value)}
          disabled={!isNew}
          placeholder="e.g. OpenAI, Claude, Local"
          className={`${inputClass} disabled:opacity-60`}
        />
        {duplicate && (
          <span className="text-xs text-destructive">A profile with this name exists.</span>
        )}
      </label>
      <label className="block space-y-1">
        <span className="text-xs font-medium">Model</span>
        <input
          value={model}
          onChange={(e) => setModel(e.target.value)}
          placeholder="openai/gpt-5, anthropic/claude-sonnet-5, ollama/llama3"
          className={inputClass}
        />
        {impliesAnthropic && (
          <span className="text-xs text-muted-foreground">
            No "/" in this name: it will be sent to Anthropic as "anthropic/{model.trim()}
            ". For another provider, prefix it, e.g. "ollama_chat/{model.trim()}" for a local Ollama
            model.
          </span>
        )}
      </label>
      <label className="block space-y-1">
        <span className="text-xs font-medium">API key</span>
        <input
          type="password"
          value={apiKey}
          onChange={(e) => setApiKey(e.target.value)}
          placeholder={profile?.apiKeySet ? "•••••••• (a key is set)" : "provider API key"}
          className={inputClass}
        />
      </label>
      <label className="block space-y-1">
        <span className="text-xs font-medium">
          Base URL
          <span className="ml-1 font-normal text-muted-foreground">(optional)</span>
        </span>
        <input
          value={baseUrl}
          onChange={(e) => setBaseUrl(e.target.value)}
          placeholder="https://… (custom or self-hosted endpoint)"
          className={inputClass}
        />
      </label>
      {error && <p className="text-xs text-destructive">{error}</p>}
      <div className="flex justify-end gap-2 pt-1">
        <Button variant="outline" onClick={onCancel}>
          Cancel
        </Button>
        <Button onClick={save} disabled={!canSave}>
          {saving ? "Saving…" : "Save"}
        </Button>
      </div>
    </div>
  )
}
