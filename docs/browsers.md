# Supported browsers

| Browser | Minimum |
| --- | --- |
| Chrome | 107 |
| Edge | 107 |
| Firefox | 104 |
| Safari | 16 |

These are the versions the app is compiled for, not an aspiration: the build emits syntax
those releases understand and no older ones do. Anything earlier gets a **blank page**, because
the browser cannot parse the bundle rather than because a feature is missing. If someone
reports a white screen with a syntax error in the console, this is why.

Internet Explorer is not supported in any version.

## Serve it over HTTPS, or lose features

Browsers withhold a set of APIs outside a [secure
context](https://developer.mozilla.org/en-US/docs/Web/Security/Secure_Contexts), and the app
uses three of them: `crypto.randomUUID`, `crypto.subtle`, and the async clipboard API. Over
plain HTTP, copy buttons stop working and anything needing Web Crypto is unavailable.

The app detects this and shows a banner that cannot be dismissed, so the cause is visible
rather than mysterious. `localhost` is exempt, so this only appears on a real
deployment. Serving over `localhost` counts as secure, so `elbi serve` runs
there; anything else needs HTTPS.

## What to check on a managed fleet

The versions above are old enough that a current browser on any platform clears them. The case
worth checking before rollout is a **centrally managed browser pinned to an extended-support
build**, which is common in hospitals and banks:

- **Firefox ESR** is fine: the oldest ESR still receiving updates is well past 104.
- **Chrome and Edge on an enterprise update policy** are fine unless the policy has frozen a
  version for more than a couple of years.
- **Safari** tracks the OS, so the constraint is really macOS 13 or later, and iOS 16 or later
  on tablets.

If your fleet is pinned below these, that is a conversation about the fleet rather than
something a configuration change here can resolve.

## Screen size

The app is built for a desktop viewport: dense tables, a notebook editor, and dashboards with
side-by-side panels. It renders on a tablet in landscape and is not designed for phones. There
is no mobile app, and no plan for one, because the work it supports is not phone work.
