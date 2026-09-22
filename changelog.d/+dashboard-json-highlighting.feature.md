The JSON a dashboard is edited through is syntax highlighted, in both the tile editor's
**JSON** tab and **Edit spec**. It is the same CodeMirror the notebook's cells use, with
bracket matching and a parse error marked in the gutter where it is, rather than
reported only when you press Save.

`useDarkTheme` moved from inside the notebook page into `hooks/useTheme`, since a second
editor needed it.
