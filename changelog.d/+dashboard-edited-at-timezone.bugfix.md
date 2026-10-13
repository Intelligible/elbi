A dashboard's last-edited time now reads in the viewer's own timezone. It was sent
without its UTC offset, so the dashboard and its exported pages showed the UTC time
labelled as local, which could put an edit hours after the snapshot that captured it.
