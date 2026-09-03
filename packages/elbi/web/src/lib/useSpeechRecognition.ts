import { useRef, useState } from "react"

// Real-time dictation via the browser's built-in Web Speech API: words stream in as
// you speak (interim results), with no model download and no network fetch. On Chrome
// the recognition runs through the platform speech service; there is nothing to load.

interface SpeechRecognitionLike {
  continuous: boolean
  interimResults: boolean
  lang: string
  start(): void
  stop(): void
  onresult: ((e: { results: ArrayLike<ArrayLike<{ transcript: string }>> }) => void) | null
  onend: (() => void) | null
  onerror: ((e: { error: string }) => void) | null
}

type Ctor = new () => SpeechRecognitionLike

function getCtor(): Ctor | null {
  if (typeof window === "undefined") return null
  const w = window as unknown as {
    SpeechRecognition?: Ctor
    webkitSpeechRecognition?: Ctor
  }
  return w.SpeechRecognition ?? w.webkitSpeechRecognition ?? null
}

export type SpeechState = "unsupported" | "idle" | "listening"

/** Stream live dictation; ``onTranscript`` fires with the full transcript so far each
time it updates (final + interim). ``onStart`` runs when listening begins. */
export function useSpeechRecognition(
  onTranscript: (transcript: string) => void,
  onStart?: () => void,
) {
  const ctor = getCtor()
  const [state, setState] = useState<SpeechState>(ctor ? "idle" : "unsupported")
  const [error, setError] = useState("")
  const rec = useRef<SpeechRecognitionLike | null>(null)

  function start() {
    if (!ctor) return
    const r = new ctor()
    r.continuous = true
    r.interimResults = true
    r.lang = "en-US"
    onStart?.()
    r.onresult = (e) => {
      let transcript = ""
      for (let i = 0; i < e.results.length; i++) {
        transcript += e.results[i][0]?.transcript ?? ""
      }
      onTranscript(transcript.trim())
    }
    r.onerror = (e) => {
      setError(e.error === "not-allowed" ? "microphone blocked" : e.error)
      setState("idle")
    }
    r.onend = () => setState("idle")
    rec.current = r
    setError("")
    r.start()
    setState("listening")
  }

  function toggle() {
    if (state === "listening") rec.current?.stop()
    else if (state === "idle") start()
  }

  return { state, error, toggle }
}
