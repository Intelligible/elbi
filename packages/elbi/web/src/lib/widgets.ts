// The live ipywidgets manager for notebooks.
//
// `HTMLManager` (from @jupyter-widgets) already loads the standard widget classes and
// renders views; on its own it is static (its comm hooks are stubs). This subclass makes
// widgets *interactive* by backing comms with the notebook's comm WebSocket: the kernel's
// comm_open/comm_msg/comm_close cross the socket into the manager, and a widget's outbound
// messages (a slider drag) go back over the socket to the kernel. Binary buffers are
// base64-encoded on the wire (the worker's relay carries no binary frames) and decoded to
// ArrayBuffers here, matching the kernel side.

import type { IClassicComm } from "@jupyter-widgets/base"
import { HTMLManager } from "@jupyter-widgets/html-manager"

export { WIDGET_MIME } from "@/lib/widget-mime"

// A comm message as it crosses the WebSocket. Kernel->browser carries `type`;
// browser->kernel carries `op`. Buffers are base64 strings either way.
type CommEnvelope = {
  type?: string
  op?: string
  content: { comm_id: string; target_name?: string; data?: unknown }
  metadata?: Record<string, unknown>
  buffers?: string[]
}

type Emit = (envelope: CommEnvelope) => void

function b64ToArrayBuffer(b64: string): ArrayBuffer {
  const binary = atob(b64)
  const bytes = new Uint8Array(binary.length)
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i)
  return bytes.buffer
}

function bufferToB64(buffer: ArrayBuffer | ArrayBufferView): string {
  const bytes =
    buffer instanceof ArrayBuffer
      ? new Uint8Array(buffer)
      : new Uint8Array(buffer.buffer, buffer.byteOffset, buffer.byteLength)
  let binary = ""
  for (let i = 0; i < bytes.length; i++) binary += String.fromCharCode(bytes[i])
  return btoa(binary)
}

let msgCounter = 0
const nextMsgId = (): string => `msg-${++msgCounter}`

// A comm bound to one widget model, relaying over the notebook's comm socket.
class WsComm implements IClassicComm {
  private msgCallback?: (x: unknown) => void
  private closeCallback?: (x: unknown) => void

  constructor(
    public comm_id: string,
    public target_name: string,
    private emit: Emit,
  ) {}

  private outbound(op: string, data: unknown, buffers?: ArrayBuffer[] | ArrayBufferView[]): string {
    const content: CommEnvelope["content"] = { comm_id: this.comm_id, data }
    if (op === "comm_open") content.target_name = this.target_name
    this.emit({ op, content, buffers: (buffers ?? []).map(bufferToB64) })
    return nextMsgId()
  }

  open(
    data: unknown,
    _callbacks?: unknown,
    _metadata?: unknown,
    buffers?: ArrayBuffer[] | ArrayBufferView[],
  ): string {
    return this.outbound("comm_open", data, buffers)
  }

  send(
    data: unknown,
    _callbacks?: unknown,
    _metadata?: unknown,
    buffers?: ArrayBuffer[] | ArrayBufferView[],
  ): string {
    return this.outbound("comm_msg", data, buffers)
  }

  close(
    data?: unknown,
    _callbacks?: unknown,
    _metadata?: unknown,
    buffers?: ArrayBuffer[] | ArrayBufferView[],
  ): string {
    return this.outbound("comm_close", data ?? {}, buffers)
  }

  on_msg(callback: (x: unknown) => void): void {
    this.msgCallback = callback
  }

  on_close(callback: (x: unknown) => void): void {
    this.closeCallback = callback
  }

  deliver(message: unknown): void {
    this.msgCallback?.(message)
  }

  finish(message: unknown): void {
    this.closeCallback?.(message)
  }
}

export class NotebookWidgetManager extends HTMLManager {
  private comms = new Map<string, WsComm>()
  private emit: Emit

  constructor(emit: Emit) {
    super()
    this.emit = emit
  }

  // A widget the frontend opens (rare in the display path); the kernel-opened case is
  // handled by `receive` below via `handle_comm_open`.
  async _create_comm(
    target_name: string,
    model_id?: string,
    data?: unknown,
    _metadata?: unknown,
    buffers?: ArrayBuffer[] | ArrayBufferView[],
  ): Promise<IClassicComm> {
    const comm = new WsComm(model_id ?? nextMsgId(), target_name, this.emit)
    this.comms.set(comm.comm_id, comm)
    if (data !== undefined) comm.open(data, undefined, undefined, buffers)
    return comm
  }

  // Route a comm message that arrived from the kernel over the socket into the widget
  // machinery: open a model, deliver an update to a model, or close one.
  receive(envelope: CommEnvelope): void {
    const kind = envelope.type ?? envelope.op
    const commId = envelope.content.comm_id
    const buffers = (envelope.buffers ?? []).map(b64ToArrayBuffer)
    const message = {
      content: envelope.content,
      metadata: envelope.metadata ?? {},
      buffers,
      // The manager reads content/metadata/buffers; a minimal header satisfies the
      // KernelMessage shape the typings expect.
      header: { msg_type: kind },
      parent_header: {},
      channel: "iopub",
    } as unknown as Parameters<HTMLManager["handle_comm_open"]>[1]

    if (kind === "comm_open") {
      const comm = new WsComm(
        commId,
        String(envelope.content.target_name ?? "jupyter.widget"),
        this.emit,
      )
      this.comms.set(commId, comm)
      void this.handle_comm_open(comm, message)
    } else if (kind === "comm_msg") {
      this.comms.get(commId)?.deliver(message)
    } else if (kind === "comm_close") {
      this.comms.get(commId)?.finish(message)
      this.comms.delete(commId)
    }
  }

  // Render the widget named by a widget-view payload into `el`.
  async renderWidget(modelId: string, el: HTMLElement): Promise<void> {
    const model = await this.get_model(modelId)
    if (!model) return
    const view = await this.create_view(model)
    await this.display_view(view, el)
  }
}

// Open the notebook's comm WebSocket and return a connected manager plus a disposer.
// The socket may connect before a kernel exists; comm traffic simply starts once a run
// creates one.
export function connectWidgets(notebookId: string): {
  manager: NotebookWidgetManager
  close: () => void
} {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:"
  const url = `${protocol}//${window.location.host}/api/notebooks/${notebookId}/comm`
  const socket = new WebSocket(url)

  const manager = new NotebookWidgetManager((envelope) => {
    if (socket.readyState === WebSocket.OPEN) socket.send(JSON.stringify(envelope))
  })

  socket.onmessage = (event) => {
    try {
      manager.receive(JSON.parse(event.data as string) as CommEnvelope)
    } catch {
      // A malformed frame is ignored rather than tearing down the widget channel.
    }
  }

  return {
    manager,
    close: () => socket.close(),
  }
}
