**Open in notebook** on a model page works for a model with no champion. It asked the
server for the training script without naming a version, so the server resolved
`@champion` — which a model whose versions are all inconclusive does not have. The
request 404'd, nothing was done with the rejection, and the button appeared dead. It
now opens the champion's script where there is one and the newest version's otherwise,
says why when a version has no script, and is not shown at all for a model with no
versions.
