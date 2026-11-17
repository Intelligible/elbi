An `elbi.yaml` or `elbi.dev.yaml` that cannot be read, such as a directory in its
place or a file that is not UTF-8, is now reported as a configuration error. Commands
that already reported invalid YAML, such as `elbi validate` and `elbi test`, now print
the same kind of error line for these files, where before they printed a traceback.
