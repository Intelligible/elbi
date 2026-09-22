The tile editor's **Derivation** and **Column** fields are pickers rather than free
text. Derivation lists what a tile may bind; Column lists the columns the bound
derivation actually returns, read from its result rather than from its serve contract,
which lists only what it chose to show. A value the list does not contain — a
derivation that has gone, a column that was renamed — stays selected and is called out,
so opening the dialog and saving never quietly clears it.

Backed by `GET /api/dashboards/columns/{name}`.
