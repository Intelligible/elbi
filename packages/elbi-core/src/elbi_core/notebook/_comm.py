"""Jupyter comm protocol for the notebook kernel worker: live ipywidgets support.

ipywidgets 8 talks to a frontend over *comms*, a bidirectional per-widget channel, and
gets them from the standalone ``comm`` package via two replaceable hooks,
``comm.create_comm`` and ``comm.get_comm_manager``. This module overwrites those hooks
so a widget's messages ride the worker's JSON relay instead of a ZMQ kernel: a
:class:`RelayComm` emits ``comm_open``/``comm_msg``/``comm_close`` on the protocol
channel, and a :class:`comm.CommManager` dispatches messages arriving back from the
frontend to the right widget.

The relay carries no binary frames, so message buffers are base64-encoded into the JSON
envelope and decoded on the way in (the convention ipywidgets' own embed path uses).
Everything is a no-op when ``comm`` is not installed, so a kernel without ipywidgets
carries no widget machinery and pays nothing.
"""

from __future__ import annotations

import base64
from collections.abc import Callable
from typing import Any

#: Called by :func:`install` for each outbound comm message; the worker points it at its
#: protocol-channel ``_send``. The message is a full envelope (``type``/``content``/
#: ``metadata``/``buffers``) shaped like a Jupyter comm message.
Sender = Callable[[dict[str, Any]], None]

#: Dispatches an inbound comm message (from the frontend) into the comm manager.
#: Returned by :func:`install`; ``None`` when ``comm`` is unavailable.
Dispatcher = Callable[[str, dict[str, Any], list[str]], None]


def install(send: Sender) -> Dispatcher | None:
    """Wire the ``comm`` package to ``send`` and return an inbound dispatcher.

    Returns ``None`` when ``comm`` is not importable (no ipywidgets in the environment),
    leaving the worker's widget support inert.
    """
    try:
        import comm
        from comm.base_comm import BaseComm, CommManager
    except ImportError:
        return None

    manager = CommManager()

    class RelayComm(BaseComm):
        """A comm whose messages are emitted on the worker's protocol channel."""

        def publish_msg(
            self,
            msg_type: str,
            data: dict[str, Any] | None = None,
            metadata: dict[str, Any] | None = None,
            buffers: list[Any] | None = None,
            **keys: Any,
        ) -> None:
            content: dict[str, Any] = {"comm_id": self.comm_id, "data": data or {}}
            if msg_type == "comm_open":
                content["target_name"] = keys.get("target_name", self.target_name)
                if keys.get("target_module"):
                    content["target_module"] = keys["target_module"]
            send(
                {
                    "type": msg_type,
                    "content": content,
                    "metadata": metadata or {},
                    "buffers": _encode_buffers(buffers),
                }
            )

    def create_comm(**kwargs: Any) -> Any:
        return RelayComm(**kwargs)

    comm.create_comm = create_comm
    comm.get_comm_manager = lambda: manager

    # Register ipywidgets' comm targets on our manager. ipywidgets only auto-registers
    # under IPython; this worker is not IPython, so it is done explicitly. Absent or
    # older ipywidgets simply means no target registration: direct display still works.
    try:
        import ipywidgets

        register = getattr(ipywidgets, "register_comm_target", None)
        if callable(register):
            register()
    except ImportError:
        pass

    def dispatch(op: str, content: dict[str, Any], buffers: list[str]) -> None:
        # ``stream``/``ident`` are the ZMQ transport args the manager ignores for our
        # relay; pass an empty ident to satisfy the signature.
        message = {"content": content, "buffers": _decode_buffers(buffers)}
        if op == "comm_open":
            manager.comm_open(None, "", message)
        elif op == "comm_msg":
            manager.comm_msg(None, "", message)
        elif op == "comm_close":
            manager.comm_close(None, "", message)

    return dispatch


def _encode_buffers(buffers: list[Any] | None) -> list[str]:
    """Base64-encode each binary buffer for the JSON envelope."""
    if not buffers:
        return []
    return [base64.b64encode(bytes(buffer)).decode("ascii") for buffer in buffers]


def _decode_buffers(buffers: list[str]) -> list[bytes]:
    """Decode base64 buffers from the frontend back to bytes."""
    return [base64.b64decode(buffer) for buffer in buffers]
